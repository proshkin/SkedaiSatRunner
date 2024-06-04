// Copyright 2010-2022 Google LLC
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdlib>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/check.h"
#include "absl/log/flags.h"
#include "absl/log/initialize.h"
#include "absl/strings/match.h"

#include "absl/strings/str_format.h" // new

#include "absl/strings/string_view.h"

#include "google/protobuf/arena.h"  // new

#include "google/protobuf/text_format.h"
#include "ortools/base/helpers.h"
#include "ortools/base/logging.h"
#include "ortools/base/options.h"
#include "ortools/base/path.h"
#include "ortools/sat/boolean_problem.h"
#include "ortools/sat/boolean_problem.pb.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/model.h"
#include "ortools/sat/opb_reader.h"
#include "ortools/sat/sat_cnf_reader.h"
#include "ortools/sat/sat_parameters.pb.h"
#include "ortools/util/file_util.h"

#include "ortools/sat/cp_model_utils.h"  // new

#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

ABSL_FLAG(
    std::string, input, "",
    "Required: input file of the problem to solve. Many format are supported:"
    ".cnf (sat, max-sat, weighted max-sat), .opb (pseudo-boolean sat/optim) "
    "and by default the CpModelProto proto (binary or text).");

ABSL_FLAG(std::string, output, "",
          "If non-empty, write the response there. By default it uses the "
          "binary format except if the file extension is '.txt'.");

ABSL_FLAG(std::string, params, "",
          "Parameters for the sat solver in a text format of the "
          "SatParameters proto, example: --params=use_conflicts:true.");

ABSL_FLAG(bool, wcnf_use_strong_slack, true,
          "If true, when we add a slack variable to reify a soft clause, we "
          "enforce the fact that when it is true, the clause must be false.");

ABSL_FLAG(std::string, callback, "",
          "If true, take the input file name and use the callback method to "
          "output the variables every change.");

ABSL_FLAG(int, step_wait, 1000,
          "Alters the initial wait before starting to print solutions "
          "if the callback flag is used.");

ABSL_FLAG(int, print_wait, 1000,
          "Alters the wait in between printing solutions after the "
          "initial solution is printed.");

ABSL_FLAG(int, objective_print_wait, 1000,
          "Alters the wait between which the current objective is printed.");

namespace operations_research {
namespace sat {
namespace {

auto startTime = std::chrono::steady_clock::now();
std::mutex mtx;
std::condition_variable cv;
bool hasNewObjective = false;
bool hasNewSolution = false;
bool hadNewSolution = false;
bool allSolutionsFound = false;
int lastStatus = -1;
long long lastObjective = -9223372036854775807LL;
std::vector<int> lastSolution;
std::chrono::time_point<std::chrono::steady_clock> lastPrintTime;
auto sec = std::chrono::seconds(1);

bool printThreadStarted = false;
std::atomic<bool> changedInLastSecond{true};
std::atomic<bool> modelSolved{false};
std::atomic<bool> variablesChanged{true};
std::atomic<int> stepWait(1);
std::atomic<int> printWait(5);
std::atomic<int> objectivePrintWait(5);


void TryToRemoveSuffix(absl::string_view suffix, std::string* str) {
  if (file::Extension(*str) == suffix) *str = file::Stem(*str);
}

std::string ExtractName(absl::string_view full_filename) {
  std::string filename = std::string(file::Basename(full_filename));
  // The order is important as '.pb.txt.gz' is a common suffix.
  TryToRemoveSuffix("gz", &filename);
  TryToRemoveSuffix("txt", &filename);
  TryToRemoveSuffix("pb", &filename);
  TryToRemoveSuffix("pbtxt", &filename);
  TryToRemoveSuffix("proto", &filename);
  TryToRemoveSuffix("prototxt", &filename);
  TryToRemoveSuffix("textproto", &filename);
  TryToRemoveSuffix("bin", &filename);
  return filename;
}

bool LoadProblem(const std::string& filename, CpModelProto* cp_model) {
  if (absl::EndsWith(filename, ".opb") ||
      absl::EndsWith(filename, ".opb.bz2")) {
    OpbReader reader;
    LinearBooleanProblem problem;
    if (!reader.Load(filename, &problem)) {
      LOG(FATAL) << "Cannot load file '" << filename << "'.";
    }
    *cp_model = BooleanProblemToCpModelproto(problem);
  } else if (absl::EndsWith(filename, ".cnf") ||
             absl::EndsWith(filename, ".cnf.xz") ||
             absl::EndsWith(filename, ".cnf.gz") ||
             absl::EndsWith(filename, ".wcnf") ||
             absl::EndsWith(filename, ".wcnf.xz") ||
             absl::EndsWith(filename, ".wcnf.gz")) {
    SatCnfReader reader(absl::GetFlag(FLAGS_wcnf_use_strong_slack));
    if (!reader.Load(filename, cp_model)) {
      LOG(FATAL) << "Cannot load file '" << filename << "'.";
    }
  } else {
    LOG(INFO) << "Reading a CpModelProto.";
    CHECK_OK(ReadFileToProto(filename, cp_model));
  }
  if (cp_model->name().empty()) {
    cp_model->set_name(ExtractName(filename));
  }
  return true;
}

std::string SolutionString(const LinearBooleanProblem& problem,
                           const std::vector<bool>& assignment) {
  std::string output;
  BooleanVariable limit(problem.original_num_variables());
  for (BooleanVariable index(0); index < limit; ++index) {
    if (index > 0) output += " ";
    absl::StrAppend(&output,
                    Literal(index, assignment[index.value()]).SignedValue());
  }
  return output;
}

void printCurrentSolution(const bool objective_only) {
  auto timeFromStart = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - startTime)
                           .count();
  std::cout << timeFromStart << "," << lastStatus << "," << lastObjective;
  if (!objective_only) {
    for (int solution : lastSolution) {
      std::cout << "," << solution;
    }
  }
  std::cout << std::endl;
}

void printObjective(const std::chrono::milliseconds& period) {
  while (true) {
    std::unique_lock<std::mutex> lock(mtx);

    if (!hasNewObjective && !allSolutionsFound) {
      cv.wait_for(lock, 60 * sec, [] { return hasNewObjective || allSolutionsFound; });
    }

    if (allSolutionsFound) {
      break;  // All solutions found and printed, exit the loop.
    }

    auto targetTime = lastPrintTime + period;
    while (hasNewObjective && std::chrono::steady_clock::now() < targetTime &&
           !allSolutionsFound) {
      cv.wait_until(lock, targetTime);  // Wait until the period expires
    }

    if (allSolutionsFound) {
      break;  // All solutions found and printed, exit the loop.
    }

    // if (hasNewObjective) {  // Print the latest solution found within the period
    printCurrentSolution(true);
    lastPrintTime = std::chrono::steady_clock::now();
    hasNewObjective = false;
    // }
  }
}

void printSolution(const std::chrono::milliseconds& period) {
  while (true) {
    std::unique_lock<std::mutex> lock(mtx);

    if (cv.wait_for(lock, period,
                    [] { return hasNewSolution || allSolutionsFound; })) {
      if (allSolutionsFound) {
        if (hadNewSolution || hasNewSolution || hasNewObjective) {
          printCurrentSolution(false);
        }
        break;
      }
      // If a new solution is found within period, reset the timer.
      hasNewSolution = false;
      hadNewSolution = true;
    } else {
      // If period elapsed without a new solution
      if (hadNewSolution) {
        printCurrentSolution(false);
        hasNewObjective = false;
        hadNewSolution = false;
      }
    }
  }
}

void copy_solution(const CpSolverResponse& r, std::vector<int>& indexes,
                   bool isAllSolutionsFound) {
  int objective = r.objective_value();
  int status = r.status();

  std::unique_lock<std::mutex> lock(mtx);
  allSolutionsFound = isAllSolutionsFound;

  if (objective > lastObjective || status != lastStatus) {
    lastObjective = objective;
    lastStatus = status;

    for (int i = 0; i < indexes.size(); i++) {
      lastSolution[i] = r.solution(indexes[i]);
    }

    hasNewSolution = true;
    hasNewObjective = true;
    cv.notify_all();
  } else if (allSolutionsFound) {
    // hadNewSolution = false;
    cv.notify_all();
  }
}


int Run() {
  SatParameters parameters;
  if (absl::GetFlag(FLAGS_input).empty()) {
    LOG(FATAL) << "Please supply a data file with --input=";
  }

  // Parse the --params flag.
  // parameters.set_log_search_progress(true);
  if (!absl::GetFlag(FLAGS_params).empty()) {
    CHECK(google::protobuf::TextFormat::MergeFromString(
        absl::GetFlag(FLAGS_params), &parameters))
        << absl::GetFlag(FLAGS_params);
  }

  // Read the problem.
  google::protobuf::Arena arena;
  // CpModelProto* cp_model = google::protobuf::Arena::CreateMessage<CpModelProto>(&arena);

  CpModelProto* cp_model = google::protobuf::Arena::Create<CpModelProto>(&arena);
  
  if (!LoadProblem(absl::GetFlag(FLAGS_input), cp_model)) {
    CpSolverResponse response;
    response.set_status(CpSolverStatus::MODEL_INVALID);
    return EXIT_SUCCESS;
  }

  Model model;
  model.Add(NewSatParameters(parameters));

  // Added code
  std::vector<int> indexes;
  bool threadStarted = false;

  stepWait.store(absl::GetFlag(FLAGS_step_wait));
  printWait.store(absl::GetFlag(FLAGS_print_wait));
  objectivePrintWait.store(absl::GetFlag(FLAGS_objective_print_wait));

  std::chrono::milliseconds objectivePeriod(objectivePrintWait);  // e.g., 500 milliseconds
  std::chrono::milliseconds solutionPeriod(printWait);  // e.g., 500 milliseconds

  lastPrintTime = std::chrono::steady_clock::now() - objectivePeriod;  // Initialize to allow immediate print

  std::thread objectivePrinter(printObjective, objectivePeriod);
  std::thread solutionPrinter(printSolution, solutionPeriod);

  if (!absl::GetFlag(FLAGS_callback).empty()) {
    std::string filename = absl::GetFlag(FLAGS_callback);

    std::ifstream file(filename);
    std::string line;

    if (file.is_open()) {
      if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
          indexes.push_back(std::stoi(value));
        }
      }
      file.close();
    } else {
      std::cerr << "Unable to open file\n";
    }
  }

  lastSolution.resize(indexes.size());

  model.Add(NewFeasibleSolutionObserver([&](const CpSolverResponse& r) {
    copy_solution(r, indexes, false);
  }));

  const CpSolverResponse response = SolveCpModel(*cp_model, &model);
  copy_solution(response, indexes, true);

  objectivePrinter.join();
  solutionPrinter.join();

  if (!absl::GetFlag(FLAGS_output).empty()) {
    if (absl::EndsWith(absl::GetFlag(FLAGS_output), "txt")) {
      CHECK_OK(file::SetTextProto(absl::GetFlag(FLAGS_output), response,
                                  file::Defaults()));
    } else {
      CHECK_OK(file::SetBinaryProto(absl::GetFlag(FLAGS_output), response,
                                    file::Defaults()));
    }
  }

  modelSolved = true;

  if (response.status() != lastStatus) {
    lastStatus = response.status();
    if (response.objective_value() != lastObjective) {
      lastObjective = response.objective_value();
      printCurrentSolution(true);
    } else {
      printCurrentSolution(false);
    }
  }

  if (response.status() == CpSolverStatus::OPTIMAL) return 10;
  if (response.status() == CpSolverStatus::FEASIBLE) return 10;
  if (response.status() == CpSolverStatus::INFEASIBLE) return 20;
  return EXIT_SUCCESS;
}

}
}
}  // namespace operations_research

static const char kUsage[] =
    "Usage: see flags.\n"
    "This program solves a given problem with the CP-SAT solver.";

int main(int argc, char** argv) {
  absl::InitializeLog();
  absl::SetProgramUsageMessage(kUsage);
  absl::ParseCommandLine(argc, argv);
  return operations_research::sat::Run();
}
