#include "../source_file_realtime_v1_async/Simulation/inc/RealTimeRestriction.h"
#include <chrono>
#include <algorithm>
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
      use_external_clock(false),
      external_clock(0.0),
      external_clock_wall_time(0.0),
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
    simulation_time.store(0.0);

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
    simulation_time.store(simulation_time.load() + simulation_step_size);
    primed.store(true);
    UpdateRestrictionLevel(GetSupervisionTime());
}

void RealTimeRestriction::NextStepWatchDog(double new_external_clock) {
    SetExternalClock(new_external_clock);
    NextStepWatchDog();
}

void RealTimeRestriction::StartWatchDog() {
    ResetClock();
    simulation_time.store(0.0);
    external_clock_wall_time.store(0.0);
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

        UpdateRestrictionLevel(GetSupervisionTime());

        if (sleep_period > 0.0) {
            std::this_thread::sleep_for(std::chrono::duration<double>(sleep_period));
        }
    }
}

void RealTimeRestriction::SetSleepPeriod(double new_sleep_period) {
    sleep_period = (new_sleep_period >= 0.0) ? new_sleep_period : 0.0;
}

void RealTimeRestriction::EnableExternalClock(double initial_clock) {
    use_external_clock.store(true);
    external_clock.store(std::max(0.0, initial_clock));
    const auto now = std::chrono::steady_clock::now();
    external_clock_wall_time.store(std::chrono::duration<double>(now - start_time).count());
}

void RealTimeRestriction::DisableExternalClock() {
    use_external_clock.store(false);
}

void RealTimeRestriction::SetExternalClock(double new_external_clock) {
    external_clock.store(std::max(0.0, new_external_clock));
    const auto now = std::chrono::steady_clock::now();
    external_clock_wall_time.store(std::chrono::duration<double>(now - start_time).count());
}

bool RealTimeRestriction::IsExternalClockEnabled() const {
    return use_external_clock.load();
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

double RealTimeRestriction::GetSupervisionTime() const {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - start_time).count();
    if (!use_external_clock.load()) {
        return elapsed;
    }

    // Match EDLUT's external-clock behavior: use the most recently supplied
    // external time and extrapolate it with the monotonic wall clock until the
    // next external update arrives.
    const double elapsed_since_update = std::max(0.0, elapsed - external_clock_wall_time.load());
    return external_clock.load() + elapsed_since_update;
}

void RealTimeRestriction::UpdateRestrictionLevel(double current_time) {
    const double gap = simulation_time.load() - current_time;
    RealTimeRestrictionLevel new_level = ALL_EVENTS_ENABLED;
    if (gap >= max_simulation_time_in_advance) {
        new_level = SIMULATION_TOO_FAST;
        restriction_level.store(new_level);
        restriction_level_counts[static_cast<std::size_t>(new_level)].fetch_add(1, std::memory_order_relaxed);
    }
    else if (gap > first_gap_time) {
        new_level = ALL_EVENTS_ENABLED;
        restriction_level.store(new_level);
        restriction_level_counts[static_cast<std::size_t>(new_level)].fetch_add(1, std::memory_order_relaxed);
    }
    else if (gap > second_gap_time) {
        new_level = LEARNING_RULES_DISABLED;
        restriction_level.store(new_level);
        restriction_level_counts[static_cast<std::size_t>(new_level)].fetch_add(1, std::memory_order_relaxed);
    }
    else if (gap > third_gap_time) {
        new_level = SPIKES_DISABLED;
        restriction_level.store(new_level);
        restriction_level_counts[static_cast<std::size_t>(new_level)].fetch_add(1, std::memory_order_relaxed);
    }
    else {
        new_level = ALL_UNESSENTIAL_EVENTS_DISABLED;
        restriction_level.store(new_level);
        restriction_level_counts[static_cast<std::size_t>(new_level)].fetch_add(1, std::memory_order_relaxed);
    }
}
