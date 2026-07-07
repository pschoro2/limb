from __future__ import annotations

from limb.embodied_learning import (
    ContactPoint,
    EmbodiedLearningConfig,
    EmbodiedMemory,
    SensorReadings,
    SensorPoint,
    TactileProfile,
    choose_action,
    read_tactile_sensors,
    should_dodge_reflex,
)


POINTY_CUE = (0.98, 0.18, 0.1, 0.96, 0.08, 0.82)
SMOOTH_CUE = (0.2, 1.0, 0.96, 0.02, 0.92, 0.08)


def test_one_painful_contact_creates_fast_visual_reflex() -> None:
    memory = EmbodiedMemory()
    sensors = read_tactile_sensors(
        sensor_points=[SensorPoint(1.0, 0.0, "finger")],
        contact_points=[ContactPoint(1.02, 0.0, sharpness_scale=1.0, area_scale=1.0)],
        tactile_profile=TactileProfile(tip_sharpness=0.96, compliance=0.08, contact_area=0.12),
    )
    reflex = choose_action(
        item_x=1.2,
        cue_vector=POINTY_CUE,
        confidence=0.88,
        sensors=sensors,
        memory=memory,
    )
    assert reflex.action == "WITHDRAW_FAST"

    memory.record_experience(POINTY_CUE, "pointy", reflex.action, sensors, timestamp=1.0)
    no_contact = read_tactile_sensors(
        sensor_points=[SensorPoint(1.0, 0.0, "finger")],
        contact_points=[ContactPoint(2.0, 0.0)],
        tactile_profile=TactileProfile(tip_sharpness=0.96, compliance=0.08, contact_area=0.12),
    )
    visual_reflex = choose_action(
        item_x=0.35,
        cue_vector=POINTY_CUE,
        confidence=0.9,
        sensors=no_contact,
        memory=memory,
    )

    assert visual_reflex.action == "WITHDRAW_FAST"
    assert "maximum avoidance" in visual_reflex.reason
    assert memory.entries[0].risk >= 0.85
    assert memory.entries[0].contacts == 1


def test_dodge_reflex_requires_more_than_ten_meters_per_second() -> None:
    memory = EmbodiedMemory()
    config = EmbodiedLearningConfig()
    painful = read_tactile_sensors(
        sensor_points=[SensorPoint(1.0, 0.0, "finger")],
        contact_points=[ContactPoint(1.02, 0.0, sharpness_scale=1.0, area_scale=1.0)],
        tactile_profile=TactileProfile(tip_sharpness=0.96, compliance=0.08, contact_area=0.12),
    )
    memory.record_experience(POINTY_CUE, "pointy", "WITHDRAW_FAST", painful, timestamp=1.0)
    slow = SensorReadings(
        contact=0.0,
        pressure=0.0,
        puncture=0.0,
        pain=0.0,
        safe_touch=False,
        painful=False,
        approaching_sensor=True,
        closing_speed=config.dodge_closing_speed_threshold,
        distance=0.7,
    )
    fast = SensorReadings(
        contact=0.0,
        pressure=0.0,
        puncture=0.0,
        pain=0.0,
        safe_touch=False,
        painful=False,
        approaching_sensor=True,
        closing_speed=config.dodge_closing_speed_threshold + 0.01,
        distance=0.7,
    )

    choice = choose_action(
        item_x=0.7,
        cue_vector=POINTY_CUE,
        confidence=0.9,
        sensors=slow,
        memory=memory,
    )

    assert choice.action == "WITHDRAW_FAST"
    assert should_dodge_reflex(choice, slow, config) is False
    assert should_dodge_reflex(choice, fast, config) is True


def test_first_pain_memory_requires_fifty_percent_pain() -> None:
    memory = EmbodiedMemory()
    below = SensorReadings(
        contact=0.6,
        pressure=0.5,
        puncture=0.5,
        pain=0.49,
        safe_touch=False,
        painful=False,
        approaching_sensor=False,
        closing_speed=0.0,
        distance=0.4,
    )
    at_threshold = SensorReadings(
        contact=0.6,
        pressure=0.5,
        puncture=0.5,
        pain=0.50,
        safe_touch=False,
        painful=True,
        approaching_sensor=False,
        closing_speed=0.0,
        distance=0.4,
    )

    warning_entry = memory.record_experience(POINTY_CUE, "pointy", "CAUTIOUS_PROBE", below, timestamp=1.0)
    assert warning_entry is not None
    assert warning_entry.risk < 0.85

    pain_entry = memory.record_experience(POINTY_CUE, "pointy", "WITHDRAW_FAST", at_threshold, timestamp=2.0)
    assert pain_entry is not None
    assert pain_entry.risk >= 0.85
    assert pain_entry.pain_peak == 0.50


def test_two_warning_contacts_cross_avoidance_threshold() -> None:
    memory = EmbodiedMemory()
    config = EmbodiedLearningConfig()
    warning = read_tactile_sensors(
        sensor_points=[SensorPoint(1.0, 0.0, "finger")],
        contact_points=[ContactPoint(1.60, 0.0, sharpness_scale=0.8, area_scale=1.0)],
        tactile_profile=TactileProfile(tip_sharpness=0.75, compliance=0.12, contact_area=0.20),
    )
    assert warning.pain >= config.warning_pain_threshold
    assert warning.pain < config.pain_threshold

    memory.record_experience(POINTY_CUE, "warning", "CAUTIOUS_PROBE", warning, timestamp=1.0)
    assert memory.entries[0].risk < config.risk_withdrawal_threshold
    memory.record_experience(POINTY_CUE, "warning", "CAUTIOUS_PROBE", warning, timestamp=2.0)

    assert memory.entries[0].risk >= config.risk_withdrawal_threshold
    assert memory.entries[0].contacts == 2


def test_low_pain_contact_creates_safe_memory() -> None:
    memory = EmbodiedMemory()
    sensors = read_tactile_sensors(
        sensor_points=[SensorPoint(1.0, 0.0, "finger")],
        contact_points=[ContactPoint(1.04, 0.0)],
        tactile_profile=TactileProfile(tip_sharpness=0.01, compliance=0.72, contact_area=0.94),
    )
    memory.record_experience(SMOOTH_CUE, "smooth", "CAUTIOUS_PROBE", sensors, timestamp=1.0)

    choice = choose_action(
        item_x=1.2,
        cue_vector=SMOOTH_CUE,
        confidence=0.84,
        sensors=sensors,
        memory=memory,
    )

    assert choice.action == "GENTLE_TOUCH"
    assert memory.entries[0].valence > 0
