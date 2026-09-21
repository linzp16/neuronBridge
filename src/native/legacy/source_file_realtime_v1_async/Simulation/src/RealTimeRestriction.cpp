#include "../source_file_realtime_v1_async/Simulation/inc/RealTimeRestriction.h"
#include <chrono>
#include <iostream>
#include <thread>

RealTimeRestriction::RealTimeRestriction()
    : simulation_step_size(0.0),
      max_simulation_time_in_advance(0.0),
      simulation_time(0.0),
      first_section(1.0),
      second_section(1.0),
      third_section(1.0),
      first_gap_time(0.0),
      second_gap_time(0.0),
      third_gap_time(0.0),
      sleep_period(0.01),
      stop_requested(false),
      started(false),
      primed(false),
      restriction_level(ALL_EVENTS_ENABLED),
      start_time(std::chrono::steady_clock::now()) {
    ResetRestrictionLevelCounts();
}

void RealTimeRestriction::SetParameterWatchDog(double new_simulation_step_size, double new_max_simulation_time_in_advance,
    float new_first_section, float new_second_section, float new_third_section) {
    simulation_step_size = (new_simulation_step_size > 0.0) ? new_simulation_step_size : 0.0;
    max_simulation_time_in_advance =
        (new_max_simulation_time_in_advance >= simulation_step_size) ? new_max_simulation_time_in_advance : simulation_step_size;
    simulation_time = 0.0;

    third_section = (new_third_section >= 0.0f && new_third_section <= 1.0f) ? new_third_section : 1.0f;
    second_section = (new_second_section >= 0.0f && new_second_section <= third_section) ? new_second_section : third_section;
    first_section = (new_first_section >= 0.0f && new_first_section <= second_section) ? new_first_section : second_section;

    first_gap_time = max_simulation_time_in_advance * (1.0 - first_section);
    second_gap_time = max_simulation_time_in_advance * (1.0 - second_section);
    third_gap_time = max_simulation_time_in_advance * (1.0 - third_section);

    stop_requested.store(false);
    started.store(false);
    primed.store(false);
    restriction_level.store(ALL_EVENTS_ENABLED);
    ResetRestrictionLevelCounts();
    ResetClock();
}

void RealTimeRestriction::NextStepWatchDog() {
    simulation_time += simulation_step_size;
    primed.store(true);
    const auto now = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = now - start_time;
    UpdateRestrictionLevel(elapsed.count());
}

void RealTimeRestriction::StartWatchDog() {
    ResetClock();
    started.store(true);
    stop_requested.store(false);
    primed.store(false);
    restriction_level.store(ALL_EVENTS_ENABLED);
    ResetRestrictionLevelCounts();
}

void RealTimeRestriction::StopWatchDog() {
    stop_requested.store(true);
}

void RealTimeRestriction::Watchdog() {
    while (!started.load() && !stop_requested.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    while (!stop_requested.load()) {
        if (!primed.load()) {
            restriction_level.store(ALL_EVENTS_ENABLED);
            if (sleep_period > 0.0) {
                std::this_thread::sleep_for(std::chrono::duration<double>(sleep_period));
            }
            continue;
        }

        const auto now = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = now - start_time;
        UpdateRestrictionLevel(elapsed.count());

        if (sleep_period > 0.0) {
            std::this_thread::sleep_for(std::chrono::duration<double>(sleep_period));
        }
    }
}

void RealTimeRestriction::SetSleepPeriod(double new_sleep_period) {
    sleep_period = (new_sleep_period >= 0.0) ? new_sleep_period : 0.0;
}

RealTimeRestrictionLevel RealTimeRestriction::GetRestrictionLevel() const {
    return restriction_level.load();
}

std::array<unsigned long long, 5> RealTimeRestriction::GetRestrictionLevelCounts() const {
    std::array<unsigned long long, 5> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = restriction_level_counts[index].load(std::memory_order_relaxed);
    }
    return result;
}

void RealTimeRestriction::ResetRestrictionLevelCounts() {
    for (std::atomic<unsigned long long>& counter : restriction_level_counts) {
        counter.store(0, std::memory_order_relaxed);
    }
}

void RealTimeRestriction::ResetClock() {
    start_time = std::chrono::steady_clock::now();
}

void RealTimeRestriction::UpdateRestrictionLevel(double current_time) {
    const double gap = simulation_time - current_time;
    RealTimeRestrictionLevel new_level = ALL_EVENTS_ENABLED;
    if (gap >= max_simulation_time_in_advance) {
        new_level = SIMULATION_TOO_FAST;
        restriction_level.store(new_level);
        restriction_level_counts[static_cast<std::size_t>(new_level)].fetch_add(1, std::memory_order_relaxed);
        std::cout << "Simulation is too fast!" << std::endl;
    }
    else if (gap > first_gap_time) {
        new_level = ALL_EVENTS_ENABLED;
        restriction_level.store(new_level);
        restriction_level_counts[static_cast<std::size_t>(new_level)].fetch_add(1, std::memory_order_relaxed);
        std::cout << "Simulation is running in real time." << std::endl;
    }
    else if (gap > second_gap_time) {
        new_level = LEARNING_RULES_DISABLED;
        restriction_level.store(new_level);
        restriction_level_counts[static_cast<std::size_t>(new_level)].fetch_add(1, std::memory_order_relaxed);
        std::cout << "Learning rules are disabled." << std::endl;
    }
    else if (gap > 0) {
        new_level = SPIKES_DISABLED;
        restriction_level.store(new_level);
        restriction_level_counts[static_cast<std::size_t>(new_level)].fetch_add(1, std::memory_order_relaxed);
        std::cout << "Spikes are disabled." << std::endl;
    }
    else {
        new_level = ALL_UNESSENTIAL_EVENTS_DISABLED;
        restriction_level.store(new_level);
        restriction_level_counts[static_cast<std::size_t>(new_level)].fetch_add(1, std::memory_order_relaxed);
        std::cout << "All unessential events are disabled." << std::endl;
    }
}
