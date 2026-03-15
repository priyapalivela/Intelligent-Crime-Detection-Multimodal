"""Data mappings for audio and text severity classification."""

AUDIO_SEVERITY_MAPPING = {
    "gun_shot":        2,
    "siren":           2,
    "drilling":        2,
    "engine_idling":   2,
    "car_horn":        1,
    "dog_bark":        1,
    "jackhammer":      1,
    "children_playing": 0,
    "street_music":    0,
    "air_conditioner": 0,
}

TEXT_SEVERITY_MAP = {
    "HOMICIDE":         2,
    "ROBBERY":          2,
    "ASSAULT":          2,
    "WEAPONS":          2,
    "BATTERY":          2,
    "THEFT":            1,
    "CRIMINAL DAMAGE":  1,
    "NARCOTICS":        1,
    "BURGLARY":         1,
    "MOTOR VEHICLE THEFT": 1,
    "OTHER OFFENSE":    0,
    "PUBLIC INDECENCY": 0,
    "DECEPTIVE PRACTICE": 0,
    "TRESPASS":         0,
}