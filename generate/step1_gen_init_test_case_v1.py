# Generate ~10,000 CCTV-style synthetic OD test cases (no doorbell views),
# with stricter physical constraints and Qwen-Image-friendly phrasing.
# Output: JSONL for Qwen-Image servers.
import json, random
from pathlib import Path

random.seed(19)

N_TOTAL = 10_000
DISTRIBUTION = {
    "positive": 3500,
    "negative": 1500,
    "false_positive_trap": 1500,
    "false_negative_risk": 1500,
    "repeat_sequence": 1000,   # frames total (≈200 seq × 5 frames)
    "edge_case": 1000,
}

_NUM_WORDS = {
    0:"zero",1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",9:"nine",
    10:"ten",11:"eleven",12:"twelve",13:"thirteen",14:"fourteen",15:"fifteen",
    16:"sixteen",17:"seventeen",18:"eighteen",19:"nineteen",20:"twenty"
}
def _to_words(n:int) -> str:
    return _NUM_WORDS.get(n, str(n))

def _pluralize(noun: str) -> str:
    """Very small pluralizer for short nouns."""
    n = noun.strip()
    lower = n.lower()
    # basic irregulars;按需扩展
    irregular = {"man":"men", "woman":"women", "person":"people", "mouse":"mice", "goose":"geese", "ox":"oxen"}
    if lower in irregular:
        out = irregular[lower]
        return out.upper() if n.isupper() else out

    def _cons(ch): return ch.isalpha() and ch.lower() not in "aeiou"

    if lower.endswith(("s","x","z","ch","sh")):
        return n + "es"
    if lower.endswith("y") and len(lower) > 1 and _cons(lower[-2]):
        return n[:-1] + "ies"
    return n + "s"

def pluralize_color_phrase(phrase: str) -> str:
    """
    Pluralize a 'color + noun' phrase while保留形容词:
    'white goat' -> 'white goats', 'red sedan' -> 'red sedans', 'SUV' -> 'SUVs'
    """
    toks = [t for t in phrase.strip().split() if t]
    if not toks:
        return "objects"
    noun = toks[-1]
    toks[-1] = _pluralize(noun)
    return " ".join(toks)


# OD classes
od_types = ["person", "animal", "vehicle", "package"]
motion_types = ["static", "non-static"]

# Vantages (neutral language: no "camera" tokens)
camera_views = [
    "fixed_eave_corner_high",     # outdoor house/business eave, ~3–4 m
    "wall_mount_mid_height",      # outdoor/indoor wall mount, ~2–2.5 m
    "pole_mount_parking_lot",     # outdoor pole in small business lot
    "indoor_ceiling_corner_high", # indoor, high-angle from ceiling corner (~3 m)
    "hallway_ceiling_mount",      # indoor hallway (~2.7 m)
    "warehouse_corner_high",      # indoor warehouse corner high (~5 m)
    "storefront_soffit_mount",    # outdoor soffit over a shop entrance (~3.5 m)
]
camera_view_text = {
    "fixed_eave_corner_high": "elevated high-angle viewpoint from a building eave",
    "wall_mount_mid_height": "mid-height wall-mounted viewpoint",
    "pole_mount_parking_lot": "elevated viewpoint from a parking-lot pole",
    "indoor_ceiling_corner_high": "high-angle viewpoint from an indoor ceiling corner",
    "hallway_ceiling_mount": "ceiling-mounted viewpoint looking down a hallway",
    "warehouse_corner_high": "high-angle viewpoint from a warehouse corner",
    "storefront_soffit_mount": "soffit-mounted viewpoint over an entrance",
}


times_of_day = ["dawn", "day", "dusk", "night"]
weathers = ["clear", "partly_cloudy", "rain_light", "rain_heavy", "fog_light", "fog_dense", "snow_light", "snow_heavy"]

# Property types & locations
property_types = ["family_house", "farm", "small_business"]

location_catalog = {
    "family_house": [
        ("residential_front_porch", "outdoor"),
        ("suburban_driveway", "outdoor"),
        ("residential_backyard", "outdoor"),
        ("garage_interior", "indoor"),
        ("house_entry_hall", "indoor"),
        ("living_room_entry", "indoor"),
        ("side_gate_path", "outdoor"),
    ],
    "farm": [
        ("farmyard_gate", "outdoor"),
        ("barn_entrance", "outdoor"),
        ("tool_shed_doorway", "outdoor"),
        ("pasture_edge", "outdoor"),
        ("tractor_parking_area", "outdoor"),
        ("farmhouse_porch", "outdoor"),
    ],
    "small_business": [
        ("shop_entrance", "outdoor"),
        ("shop_front_counter", "indoor"),
        ("backroom_storage", "indoor"),
        ("warehouse_loading_dock", "outdoor"),
        ("parking_lot_small", "outdoor"),
        ("office_hallway", "indoor"),
        ("cafe_entrance", "outdoor"),
        ("warehouse_interior_aisle", "indoor"),
    ],
}

# Weather → textual
wmap = {
    "clear":"clear weather",
    "partly_cloudy":"partly cloudy sky",
    "rain_light":"light rain, damp surfaces",
    "rain_heavy":"heavy rain with wet reflective ground",
    "fog_light":"light fog haze reducing contrast",
    "fog_dense":"dense fog with low visibility",
    "snow_light":"light snowfall, scattered accumulation",
    "snow_heavy":"heavy snowfall with accumulation"
}
# Time-of-day → textual
tod_map = {
    "dawn": "soft dawn light with long shadows",
    "day": "bright daylight with natural shadows",
    "dusk": "warm dusk light, long shadows",
    "night": "night lighting from porch or street lamps; IR night possible"
}

# Lighting conditions
lighting_conditions = [
    "normal_daylight",
    "overcast_diffuse",
    "backlit_low_sun",
    "porch_light_only",
    "streetlight_sodium",
    "floodlight_on",
    "indoor_fluorescent",
    "warehouse_metal_halide",
    "ir_night_monochrome"
]

# Visual attributes
distances = ["near", "mid", "far"]
occlusion_levels = [0, 25, 50, 75, 90]
sensor_artifacts = ["motion_blur", "image_noise", "low_resolution", "rolling_shutter_skew", None]
lens_effects = [None, "slight_vignetting", "minor_chromatic_aberration"]

# Colors/phrasing
colors_people = ["red jacket", "blue coat", "black hoodie", "grey clothing", "high-visibility vest", "white shirt"]
colors_vehicle = ["blue SUV", "red sedan", "white van", "grey truck", "black motorcycle"]
colors_animal_house = ["brown dog", "black cat"]
colors_animal_farm  = ["white goat", "speckled cow", "red fox", "chicken"]
colors_package = ["brown cardboard box", "white parcel", "yellow padded mailer"]
package_places_outdoor = ["on doorstep", "near the front door", "on doormat", "beside the mailbox", "under porch bench"]
package_places_indoor  = ["by the entry mat", "beside the reception desk", "just inside the doorway"]

# Confusers for FP traps
confusers = {
    "person": ["mannequin", "statue", "scarecrow", "cardboard cutout", "coat rack with a long coat", "shadow shaped like a person"],
    "animal": ["stuffed toy dog", "garden gnome", "animal statue", "fur coat on a chair", "blowing plastic bag"],
    "vehicle": ["shopping cart", "wheelbarrow", "ride-on lawn mower", "dumpster"],
    "package": ["doormat", "trash bag", "shoe box lid", "folded cardboard", "plant pot", "stack of newspapers"]
}

# Global hard negative (moved here, not in the positive prefix)
GLOBAL_NEGATIVE_HARD = (
    "camera, cameras, security camera, CCTV, surveillance camera, dome camera, "
    "visible camera hardware, device, lens, fisheye, fisheye edges, "
    "webcam, gopro, tripod, photographer, selfie, reflection of a camera, "
    "recording light, timestamp overlay, timecode overlay, HUD, UI overlay, text watermark"
)

STYLE_PROFILE = "modern_clean"  # 可选: "modern_clean" | "legacy_dvr" | "night_ir"

def _style_base():
    # 基础摄影语气：真实感 + 光影自然；不带任何分辨率/画幅数字
    return ["photorealistic", "realistic lighting", "natural color balance", "high detail", "sharp focus"]

def _style_monitor_like_common():
    # 共同的“监控风味”特征（轻量，避免过度劣化）
    return ["fixed high-angle viewpoint", "small-sensor video aesthetic", "mild compression artifacts", "modest sharpening halos"]

def _style_modern_clean(io, tod, lighting):
    parts = _style_base() + _style_monitor_like_common()
    if tod == "night":
        parts += ["low-light video look", "subtle shadow noise", "soft denoise"]
    if lighting == "ir_night_monochrome":
        parts += ["monochrome infrared appearance", "enhanced scene contrast"]
    if io == "outdoor" and lighting in ["streetlight_sodium", "porch_light_only"]:
        parts += ["slight bloom around lamps", "warm ambient cast"]
    if io == "indoor" and lighting == "indoor_fluorescent":
        parts += ["neutral-to-cool fluorescent cast"]
    return parts

def _style_legacy_dvr(io, tod, lighting):
    parts = _style_base() + _style_monitor_like_common()
    parts += ["softer micro-contrast", "slightly raised blacks", "coarser grain structure"]
    if tod == "night":
        parts += ["visible luminance noise", "gentle motion smearing"]
    if lighting == "ir_night_monochrome":
        parts += ["monochrome infrared appearance", "specular highlights on reflective surfaces"]
    return parts

def _style_night_ir(io, tod, lighting):
    parts = _style_base() + _style_monitor_like_common()
    if tod == "night":
        parts += ["low-light video look", "subtle sensor noise", "slight motion blur"]
    if lighting == "ir_night_monochrome":
        parts += ["monochrome infrared appearance", "crisp edges with high local contrast"]
    return parts

def style_prefix_for(io: str, tod: str, lighting: str) -> str:
    if STYLE_PROFILE == "legacy_dvr":
        parts = _style_legacy_dvr(io, tod, lighting)
    elif STYLE_PROFILE == "night_ir":
        parts = _style_night_ir(io, tod, lighting)
    else:  # "modern_clean"
        parts = _style_modern_clean(io, tod, lighting)
    # 去重并拼接
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            out.append(p); seen.add(p)
    return ", ".join(out)


def choose_lighting(tod, io):
    """Tie lighting to time-of-day and indoor/outdoor realism."""
    if io == "indoor":
        # No outdoor lamps/weather-only lighting indoors
        return random.choice(["indoor_fluorescent", "normal_daylight", "overcast_diffuse"])
    # Outdoor
    if tod == "night":
        return random.choices(
            ["porch_light_only","streetlight_sodium","floodlight_on","ir_night_monochrome"],
            weights=[3,2,2,3]
        )[0]
    if tod in ["dawn","dusk"]:
        return random.choice(["backlit_low_sun","overcast_diffuse","normal_daylight"])
    return random.choice(["normal_daylight","overcast_diffuse"])

def env_negative_tokens(io, tod):
    tokens = []
    # Keep class-agnostic environmental consistency negatives
    if io == "indoor":
        tokens += ["sky", "clouds", "rain streaks on lens", "snow falling", "street road scene"]
    else:
        tokens += ["indoor ceiling tiles", "office furniture cluster (when outdoor)"]
    if tod == "night":
        tokens += ["strong bright sun beams", "harsh midday shadows"]
    else:
        tokens += ["IR night monochrome look"]
    return tokens

def negative_prompt_base(forbid_primary=None, io=None, tod=None):
    np = [
        "cartoon", "illustration", "CGI", "render", "deformed", "low-quality",
        "text", "watermark", "timestamp", "date overlay", "HUD", "UI overlay",
        # suppress any accidental text/number artifacts
        "captions", "subtitles", "on-screen text", "numbers", "digits",
        "numeric overlays", "alphanumeric characters",
    ]
    if forbid_primary:
        if forbid_primary == "person":
            np += ["person", "people", "human", "man", "woman"]
        elif forbid_primary == "animal":
            np += ["animal", "dog", "cat", "cow", "goat", "fox", "chicken", "animals"]
        elif forbid_primary == "vehicle":
            np += ["car", "truck", "van", "motorcycle", "vehicle", "vehicles"]
        elif forbid_primary == "package":
            np += ["package", "parcel", "box", "delivery"]

    # environment consistency
    if io or tod:
        np += env_negative_tokens(io, tod)

    # global hard negative (device/fisheye/etc.)
    np.append(GLOBAL_NEGATIVE_HARD)

    # de-dup
    seen, clean = set(), []
    for x in np:
        if x and x not in seen:
            clean.append(x); seen.add(x)
    return ", ".join(clean)


def background_prompt(property_type, location, io, tod, weather, lighting):
    mapping = {
        "residential_front_porch": "residential front porch with door and steps",
        "suburban_driveway": "suburban driveway facing the street from a garage corner",
        "residential_backyard": "residential backyard with patio and lawn",
        "garage_interior": "home garage interior with shelves and stored items",
        "house_entry_hall": "house entry hallway with coat rack and entry mat",
        "living_room_entry": "living room entry seen from hallway",
        "side_gate_path": "side path along a fence leading to a gate",
        "farmyard_gate": "farmyard gate with fences and a gravel path",
        "barn_entrance": "barn entrance with wooden doors and hay bales nearby",
        "tool_shed_doorway": "tool shed doorway with tools and equipment",
        "pasture_edge": "pasture edge with grass field and fence line",
        "tractor_parking_area": "parking area for farm vehicles on gravel ground",
        "farmhouse_porch": "farmhouse porch with wooden steps and rails",
        "shop_entrance": "shop entrance with glass door and signage",
        "shop_front_counter": "shop front counter area with shelves behind",
        "backroom_storage": "backroom storage with boxes and metal racks",
        "warehouse_loading_dock": "small loading dock with roll-up door and pallets",
        "parking_lot_small": "small parking lot with painted lines",
        "office_hallway": "office hallway corridor with doors and carpet",
        "cafe_entrance": "café entrance with awning and outdoor seating",
        "warehouse_interior_aisle": "warehouse interior aisle with tall shelving",
    }
    parts = [mapping.get(location, "generic property scene")]
    parts.append("outdoor" if io == "outdoor" else "indoor")
    parts.append(tod_map[tod])
    if io == "outdoor":
        parts.append(wmap[weather])
    else:
        parts.append("no direct weather visible")
    prop_text = {"family_house":"residential setting","farm":"farm setting","small_business":"small business setting"}[property_type]
    parts.append(prop_text)
    # lighting
    lit_text = {
        "ir_night_monochrome": "infrared night mode, monochrome look with IR reflections",
        "streetlight_sodium": "streetlight casts warm sodium-like glow",
        "porch_light_only": "only porch light illuminates the area",
        "floodlight_on": "bright floodlight illuminates the scene",
        "indoor_fluorescent": "indoor fluorescent lighting",
        "warehouse_metal_halide": "metal-halide high-bay lights",
        "backlit_low_sun": "low sun backlighting",
        "overcast_diffuse": "diffuse soft lighting",
        "normal_daylight": "normal lighting levels"
    }.get(lighting, "normal lighting levels")
    parts.append(lit_text)
    return ", ".join(parts)

def comp_guidance(camera_view, distance, occl, extras=None):
    parts = [camera_view_text[camera_view]]
    if distance == "near": parts.append("subject occupies a large portion of the frame")
    if distance == "mid":  parts.append("subject at mid distance with clear detail")
    if distance == "far":  parts.append("subject smaller in frame with full context")

    if occl >= 50:
        parts.append("subject heavily occluded by foreground elements")
    elif occl >= 25:
        parts.append("subject partially occluded")

    if extras:
        parts.append(extras)
    parts.append("natural perspective")
    return ", ".join(parts)


def person_appearance_phrase():
    c = random.choice(colors_people)
    wearable = ["hoodie","coat","jacket","vest","shirt"]
    for w in wearable:
        if w in c:
            return f"wearing a {c}"
    return f"wearing {c}"

def pick_color_for_class(od, property_type=None):
    if od == "person":
        return person_appearance_phrase()
    if od == "vehicle":
        return random.choice(colors_vehicle)
    if od == "animal":
        # bias animal colors by property type
        if property_type == "farm":
            return random.choice(colors_animal_farm)
        else:
            # houses/small business: mostly pets or urban wildlife
            return random.choice(colors_animal_house + ["red fox"])
    if od == "package":
        return random.choice(colors_package)
    return ""

# --- Realism constraints -----------------------------------------------------

def sample_property_type(od):
    # Bias by class to avoid odd placements
    if od == "animal":
        weights = {"family_house": 0.35, "farm": 0.55, "small_business": 0.10}
    elif od == "vehicle":
        weights = {"family_house": 0.45, "farm": 0.15, "small_business": 0.40}
    elif od == "package":
        weights = {"family_house": 0.55, "farm": 0.05, "small_business": 0.40}
    else:  # person
        weights = {"family_house": 0.45, "farm": 0.15, "small_business": 0.40}
    r = random.random(); cum = 0.0
    for pt, w in weights.items():
        cum += w
        if r <= cum:
            return pt
    return "family_house"

def allowed_locations_for(od, property_type):
    """Return a subset of locations that are plausible for the given class."""
    locs = location_catalog[property_type]
    plausible = []
    for loc, io in locs:
        if od == "vehicle":
            # vehicles: driveways, parking, loading docks, garage/warehouse only
            if (io == "outdoor" and loc in ["suburban_driveway","parking_lot_small","warehouse_loading_dock","tractor_parking_area"]) or \
               (io == "indoor" and loc in ["garage_interior","warehouse_interior_aisle"]):
                plausible.append((loc, io))
        elif od == "package":
            # entries/hallways/porches; indoor entries at business
            if loc in ["residential_front_porch","shop_entrance","house_entry_hall","office_hallway","cafe_entrance","farmhouse_porch"]:
                plausible.append((loc, io))
        elif od == "animal":
            # farms, yards, parking edge sometimes; avoid office interiors
            if not (io == "indoor" and loc in ["office_hallway","shop_front_counter","backroom_storage","house_entry_hall","living_room_entry"]):
                plausible.append((loc, io))
        else:  # person
            plausible.append((loc, io))
    # Fallback to all if filtering wiped out (should be rare)
    return plausible if plausible else locs

def sample_location_for(od, property_type):
    candidates = allowed_locations_for(od, property_type)
    loc, io = random.choice(candidates)
    return loc, io

def sample_camera_view(io, property_type, location):
    if io == "indoor":
        if "hallway" in location:
            return "hallway_ceiling_mount"
        if "warehouse" in location:
            return "warehouse_corner_high"
        return "indoor_ceiling_corner_high"
    # outdoor
    if location in ["parking_lot_small","warehouse_loading_dock","tractor_parking_area"]:
        return "pole_mount_parking_lot"
    if location in ["shop_entrance","cafe_entrance","residential_front_porch","farmhouse_porch"]:
        return "storefront_soffit_mount" if "shop" in location or "cafe" in location else "fixed_eave_corner_high"
    return random.choice(["fixed_eave_corner_high","wall_mount_mid_height"])

def assemble_prompt(bg, pos, comp, io=None, tod=None, lighting=None):
    return f"{style_prefix_for(io, tod, lighting)}. Scene: {bg}. Primary: {pos}. Framing and angle: {comp}."


cases = []

# --- Generators ---------------------------------------------------------------

def gen_positive(n, start_idx):
    for k in range(n):
        i = start_idx + k
        od = random.choice(od_types)
        property_type = sample_property_type(od)
        loc, io = sample_location_for(od, property_type)
        camera_view = sample_camera_view(io, property_type, loc)
        motion = random.choice(motion_types)
        tod = random.choices(times_of_day, weights=[1,3,1,1])[0]
        weather = random.choices(weathers, weights=[5,4,2,1,1,1,1,1])[0] if io=="outdoor" else "n/a"
        lighting = choose_lighting(tod, io)
        # distance/occlusion correlation: far → higher chance of occlusion being low; near → more occlusion risk
        distance = random.choices(distances, weights=[2,3,2])[0]
        occl = random.choices(occlusion_levels, weights=[4,3,2,1,1] if distance!="near" else [3,3,2,1,1])[0]
        lens = random.choice(lens_effects)
        artifact = random.choice(sensor_artifacts)
        color = pick_color_for_class(od, property_type)

        # class-specific subject phrasing for realism
        if od == "person":
            where = "near the doorway" if "entrance" in loc or "porch" in loc else ("on the walkway" if "path" in loc or "hallway" in loc else "in the scene")
            verb = "walking" if motion=="non-static" else "standing"
            count = 1 if random.random() < 0.9 else random.randint(2, 4)
            if count == 1:
                subject = f"one person {person_appearance_phrase()} {verb} {where}"
            else:
                subject = f"{_to_words(count)} people {verb} {where}"
        elif od == "animal":
            where = "by the fence" if "pasture" in loc or "farm" in loc or "yard" in loc else "near the edge of the scene"
            verb = "moving" if motion=="non-static" else "still"
            count = 1 if property_type != "farm" else (1 if random.random()<0.8 else random.randint(2,5))
            color = pick_color_for_class(od, property_type)
            if count == 1:
                subject = f"one {color} {verb} {where}"
            else:
                subject = f"{_to_words(count)} {pluralize_color_phrase(color)} {verb} {where}"
        # crude pluralization
        elif od == "vehicle":
            where = "on the driveway" if "driveway" in loc else ("in a parking space" if "parking" in loc else ("at the loading area" if "loading_dock" in loc else "on the paved area"))
            verb = "driving past" if motion=="non-static" else "parked"
            count = 1 if random.random() < 0.97 else 2
            color = pick_color_for_class(od, property_type)
            if count == 1:
                subject = f"one {color} {verb} {where}"
            else:
                subject = f"{_to_words(count)} {pluralize_color_phrase(color)} {verb} {where}"
        else:  # package
            place = random.choice(package_places_outdoor if io=="outdoor" else package_places_indoor)
            subject = f"one {pick_color_for_class('package')} placed {place}"
            count = 1

        extras = []
        if lens and random.random() < 0.20:
            extras.append(lens.replace("_", " "))
        if artifact and random.random() < 0.25:
            extras.append(artifact.replace("_", " "))
        comp = comp_guidance(camera_view, distance, occl, ", ".join(extras) if extras else None)

        bg = background_prompt(property_type, loc, io, tod, weather, lighting)

        data = {
            "test_case_id": f"OD-POS-{i:05d}",
            "scenario_category": "positive",
            "test_subcategory": "baseline",
            "property_type": property_type,
            "od_type_primary": od,
            "od_state": "enabled",
            "motion_type": motion,
            "attributes": {
                "indoor_outdoor": io,
                "location": loc,
                "time_of_day": tod,
                "weather": weather if io=="outdoor" else "n/a",
                "lighting": lighting,
                "camera_view": camera_view,
                "distance": distance,
                "occlusion_pct": occl,
                "lens_effect": lens,
                "sensor_artifact": artifact,
                "object_count": count
            },
            "background_prompt": bg,
            "positive_prompt": subject,
            "negative_prompt": negative_prompt_base(io=io, tod=tod),
            "composition_guidance": comp,
            "prompt": assemble_prompt(bg, subject, comp, io=io, tod=tod, lighting=lighting),
            "seed": random.randint(1, 2_000_000_000),
            "expected_detection": {k: (k==od) for k in od_types},  # only primary expected
            "risk_tags": ["baseline_positive"]
        }
        cases.append(data)

def gen_negative(n, start_idx):
    for k in range(n):
        i = start_idx + k
        od = random.choice(od_types)  # class under test
        property_type = sample_property_type(od)
        loc, io = sample_location_for(od, property_type)
        camera_view = sample_camera_view(io, property_type, loc)
        tod = random.choice(times_of_day)
        weather = random.choice(weathers) if io=="outdoor" else "n/a"
        lighting = choose_lighting(tod, io)
        distance = random.choice(distances)
        occl = random.choice(occlusion_levels)

        bg = background_prompt(property_type, loc, io, tod, weather, lighting)

        # Either empty scene or other non-target present; od_state may be enabled or disabled,
        # but expected detection for the primary under test is False
        if random.random() < 0.55:
            pos_prompt = "no primary objects; natural background details only"
        else:
            other_od = random.choice([x for x in od_types if x != od])
            other_color = pick_color_for_class(other_od, property_type)
            if other_od == "person":
                pos_prompt = f"one person {person_appearance_phrase()} passing through"
            elif other_od == "vehicle":
                pos_prompt = f"one {other_color} driving past on a nearby road"
            elif other_od == "animal":
                pos_prompt = f"one {other_color} moving near the fence line"
            else:
                place = random.choice(package_places_outdoor if io=="outdoor" else package_places_indoor)
                pos_prompt = f"one {other_color} {place}"

        comp = comp_guidance(camera_view, distance, occl)
        neg_prompt = negative_prompt_base(forbid_primary=od, io=io, tod=tod)

        data = {
            "test_case_id": f"OD-NEG-{i:05d}",
            "scenario_category": "negative",
            "test_subcategory": "no_detection_expected",
            "property_type": property_type,
            "od_type_primary": od,
            "od_state": "enabled" if random.random() < 0.6 else "disabled",
            "motion_type": random.choice(motion_types),
            "attributes": {
                "indoor_outdoor": io,
                "location": loc,
                "time_of_day": tod,
                "weather": weather if io=="outdoor" else "n/a",
                "lighting": lighting,
                "camera_view": camera_view,
                "distance": distance,
                "occlusion_pct": occl,
                "object_count": 0
            },
            "background_prompt": bg,
            "positive_prompt": pos_prompt,
            "negative_prompt": neg_prompt,
            "composition_guidance": comp,
            "prompt": assemble_prompt(bg, pos_prompt, comp, io=io, tod=tod, lighting=lighting),
            "seed": random.randint(1, 2_000_000_000),
            "expected_detection": {k: False for k in od_types},
            "risk_tags": ["negative_no_target"]
        }
        cases.append(data)

def gen_false_positive_trap(n, start_idx):
    for k in range(n):
        i = start_idx + k
        od = random.choice(od_types)
        property_type = sample_property_type(od)
        loc, io = sample_location_for(od, property_type)
        camera_view = sample_camera_view(io, property_type, loc)
        tod = random.choice(times_of_day)
        weather = random.choice(weathers) if io=="outdoor" else "n/a"
        lighting = choose_lighting(tod, io)
        distance = random.choice(distances)
        occl = random.choice(occlusion_levels)

        conf = random.choice(confusers[od])
        bg = background_prompt(property_type, loc, io, tod, weather, lighting)
        pos_prompt = f"realistic {conf} positioned naturally in scene; no actual {od} present"
        comp = comp_guidance(camera_view, distance, occl)
        neg_prompt = negative_prompt_base(forbid_primary=od, io=io, tod=tod)

        data = {
            "test_case_id": f"OD-FP-{i:05d}",
            "scenario_category": "false_positive_trap",
            "test_subcategory": "confuser_object",
            "property_type": property_type,
            "od_type_primary": od,
            "od_state": "enabled",
            "motion_type": "static",
            "attributes": {
                "indoor_outdoor": io,
                "location": loc,
                "time_of_day": tod,
                "weather": weather if io=="outdoor" else "n/a",
                "lighting": lighting,
                "camera_view": camera_view,
                "distance": distance,
                "occlusion_pct": occl,
                "object_count": 0,
                "confuser": conf
            },
            "background_prompt": bg,
            "positive_prompt": pos_prompt,
            "negative_prompt": neg_prompt,
            "composition_guidance": comp,
            "prompt": assemble_prompt(bg, pos_prompt, comp, io=io, tod=tod, lighting=lighting),
            "seed": random.randint(1, 2_000_000_000),
            "expected_detection": {k: False for k in od_types},
            "risk_tags": ["false_positive_risk"]
        }
        cases.append(data)

def gen_false_negative_risk(n, start_idx):
    for k in range(n):
        i = start_idx + k
        od = random.choice(od_types)
        property_type = sample_property_type(od)
        loc, io = sample_location_for(od, property_type)
        camera_view = sample_camera_view(io, property_type, loc)
        motion = random.choice(motion_types)
        tod = random.choice(times_of_day)
        weather = random.choice(["fog_dense","rain_heavy","snow_heavy"] if (io=="outdoor" and random.random()<0.7) else (weathers if io=="outdoor" else ["n/a"]))
        lighting = choose_lighting(tod, io)
        distance = random.choice(["far","mid"])
        occl = random.choice([50,75,90])
        artifact = random.choice(["motion_blur","low_resolution","image_noise","rolling_shutter_skew"])

        bg = background_prompt(property_type, loc, io, tod, weather, lighting)
        color = pick_color_for_class(od, property_type)
        if od == "person":
            subject = f"one person {color} partly hidden near a fence or doorway"
        elif od == "animal":
            subject = f"one {color} partly behind bushes or fencing"
        elif od == "vehicle":
            subject = f"one {color} mostly hidden behind another parked vehicle"
        else:
            place = ("under a bench in shadow" if io=="outdoor" else "in a shaded floor corner")
            subject = f"one {color} tucked {place}"
        if motion == "non-static":
            subject += ", slight motion visible"

        comp = comp_guidance(camera_view, distance, occl, extras=f"pronounced {artifact.replace('_',' ')}")

        data = {
            "test_case_id": f"OD-FN-{i:05d}",
            "scenario_category": "false_negative_risk",
            "test_subcategory": "low_visibility_or_occlusion",
            "property_type": property_type,
            "od_type_primary": od,
            "od_state": "enabled",
            "motion_type": motion,
            "attributes": {
                "indoor_outdoor": io,
                "location": loc,
                "time_of_day": tod,
                "weather": weather if io=="outdoor" else "n/a",
                "lighting": lighting,
                "camera_view": camera_view,
                "distance": distance,
                "occlusion_pct": occl,
                "sensor_artifact": artifact,
                "object_count": 1
            },
            "background_prompt": bg,
            "positive_prompt": subject,
            "negative_prompt": negative_prompt_base(io=io, tod=tod),
            "composition_guidance": comp,
            "prompt": assemble_prompt(bg, subject, comp, io=io, tod=tod, lighting=lighting),
            "seed": random.randint(1, 2_000_000_000),
            "expected_detection": {k: (k==od) for k in od_types},
            "risk_tags": ["false_negative_risk", "hard_conditions"]
        }
        cases.append(data)

def gen_edge_cases(n, start_idx):
    edge_subs = [
        ("extreme_backlight","strong backlight with mild lens flare"),
        ("lens_obstruction","raindrops, dirt smears, or spider webs near lens"),
        ("headlights_glare","vehicle headlights toward the viewpoint, glare on wet pavement"),
        ("corner_entry","subject entering from extreme frame corner"),
        ("extreme_closeup","object very close, partial view at frame edges"),
        ("crowded_scene","many overlapping people/objects"),
        ("reflection_trick","strong reflection in glass door/window mimicking object"),
        ("insect_close_to_lens","flying insect close to lens, blurred"),
        ("snowflakes_near_lens","snowflakes near lens, bokeh-like spots"),
    ]
    for k in range(n):
        i = start_idx + k
        od = random.choice(od_types)
        property_type = sample_property_type(od)
        loc, io = sample_location_for(od, property_type)
        camera_view = sample_camera_view(io, property_type, loc)
        motion = random.choice(motion_types)
        tod = random.choice(times_of_day)
        weather = random.choice(weathers) if io=="outdoor" else "n/a"
        lighting = choose_lighting(tod, io)
        distance = random.choice(distances)
        occl = random.choice(occlusion_levels)
        sub = random.choice(edge_subs)

        bg = background_prompt(property_type, loc, io, tod, weather, lighting)
        color = pick_color_for_class(od, property_type)
        if od == "person":
            subject = f"one person {color} {'moving' if motion=='non-static' else 'standing'}"
        elif od == "animal":
            subject = f"one {color} {'running' if motion=='non-static' else 'still'}"
        elif od == "vehicle":
            subject = f"one {color} {'moving' if motion=='non-static' else 'parked'}"
        else:
            subject = f"one {color} on the ground"
        comp = comp_guidance(camera_view, distance, occl, extras=sub[1])
        if sub[0] == "crowded_scene" and od == "person":
            count = random.randint(5, 20)
            subject = subject.replace("one person", f"{_to_words(count)} people")


        data = {
            "test_case_id": f"OD-EDGE-{i:05d}",
            "scenario_category": "edge_case",
            "test_subcategory": sub[0],
            "property_type": property_type,
            "od_type_primary": od,
            "od_state": "enabled",
            "motion_type": motion,
            "attributes": {
                "indoor_outdoor": io,
                "location": loc,
                "time_of_day": tod,
                "weather": weather if io=="outdoor" else "n/a",
                "lighting": lighting,
                "camera_view": camera_view,
                "distance": distance,
                "occlusion_pct": occl
            },
            "background_prompt": bg,
            "positive_prompt": subject,
            "negative_prompt": negative_prompt_base(io=io, tod=tod),
            "composition_guidance": comp,
            "prompt": assemble_prompt(bg, subject, comp, io=io, tod=tod, lighting=lighting),
            "seed": random.randint(1, 2_000_000_000),
            "expected_detection": {k: (k==od) for k in od_types},
            "risk_tags": ["edge_case", sub[0]]
        }
        cases.append(data)

def gen_repeat_sequences(n_frames_total, start_idx):
    frames_per_seq = 5
    n_seq = max(1, n_frames_total // frames_per_seq)
    frame_i = start_idx
    for s in range(n_seq):
        seq_id = f"SEQ-{s+1:04d}"
        od = random.choice(od_types)
        property_type = sample_property_type(od)
        loc, io = sample_location_for(od, property_type)
        camera_view = sample_camera_view(io, property_type, loc)
        tod = random.choice(times_of_day)
        weather = random.choice(weathers) if io=="outdoor" else "n/a"
        lighting = choose_lighting(tod, io)
        pattern = random.choice([
            "enter_stop_exit",
            "enter_exit_reenter",
            "multi_passes",
            "linger_by_door",
            "parking_lot_loops",
            "animal_in_and_out",
            "package_drop_and_removal"
        ])
        occl_base = random.choice([0,25,50])

        for f in range(frames_per_seq):
            distance = random.choice(distances)
            occl = min(90, occl_base + (5 if pattern in ["enter_stop_exit","enter_exit_reenter","parking_lot_loops"] else 0)*f)
            motion = "non-static" if pattern not in ["enter_stop_exit","package_drop_and_removal"] or f in [0,4] else ("static" if f in [2] else "non-static")
            bg = background_prompt(property_type, loc, io, tod, weather, lighting)

            color = pick_color_for_class(od, property_type)
            if pattern == "package_drop_and_removal":
                od = "package"  # force package sequence
            if od == "person":
                subject = f"one person {color} {'walking' if motion=='non-static' else 'waiting'}"
                if pattern == "linger_by_door" and f in [1,2,3]:
                    subject += " near the doorway"
            elif od == "animal":
                subject = f"one {color} {'moving' if motion=='non-static' else 'still'}"
                if pattern == "animal_in_and_out":
                    subject += ", repeatedly entering and leaving frame"
            elif od == "vehicle":
                subject = f"one {color} {'moving across frame' if motion=='non-static' else 'stopped briefly'}"
                if pattern == "parking_lot_loops":
                    subject += ", looping across parking lot lanes"
            else:
                subject = f"one {color} {'newly placed' if f==1 else ('still there' if f in [2,3] else 'removed')} on the ground"

            comp = comp_guidance(camera_view, distance, occl)
            if pattern == "enter_exit_reenter" and f == 3:
                comp += ", subject re-enters frame"
            if pattern == "multi_passes":
                comp += ", subject passes across frame multiple times"

            data = {
                "test_case_id": f"OD-REP-{frame_i:05d}",
                "scenario_category": "repeat_sequence",
                "test_subcategory": pattern,
                "sequence_id": seq_id,
                "sequence_frame_index": f,
                "property_type": property_type,
                "od_type_primary": od,
                "od_state": "enabled",
                "motion_type": motion,
                "attributes": {
                    "indoor_outdoor": io,
                    "location": loc,
                    "time_of_day": tod,
                    "weather": weather if io=="outdoor" else "n/a",
                    "lighting": lighting,
                    "camera_view": camera_view,
                    "distance": distance,
                    "occlusion_pct": occl,
                    "object_count": 1
                },
                "background_prompt": bg,
                "positive_prompt": subject,
                "negative_prompt": negative_prompt_base(io=io, tod=tod),
                "composition_guidance": comp,
                "prompt": assemble_prompt(bg, subject, comp, io=io, tod=tod, lighting=lighting),
                "seed": random.randint(1, 2_000_000_000),
                "expected_detection": {k: (k==od) for k in od_types},
                "risk_tags": ["repeat_duplicate_notifications"]
            }
            cases.append(data)
            frame_i += 1

# --- Generate according to distribution --------------------------------------

idx = 1
gen_positive(DISTRIBUTION["positive"], idx); idx += DISTRIBUTION["positive"]
gen_negative(DISTRIBUTION["negative"], idx); idx += DISTRIBUTION["negative"]
gen_false_positive_trap(DISTRIBUTION["false_positive_trap"], idx); idx += DISTRIBUTION["false_positive_trap"]
gen_false_negative_risk(DISTRIBUTION["false_negative_risk"], idx); idx += DISTRIBUTION["false_negative_risk"]
gen_edge_cases(DISTRIBUTION["edge_case"], idx); idx += DISTRIBUTION["edge_case"]
gen_repeat_sequences(DISTRIBUTION["repeat_sequence"], idx); idx += DISTRIBUTION["repeat_sequence"]

# Shuffle for variety
random.shuffle(cases)

# Write JSONL (versioned filename)
out_path = Path("../dataset/od_synth_cases_10000_cctv_v1_.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    for row in cases:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
