# Generate ~10,000 CCTV-style synthetic OD test cases (no doorbell views),
# focused on family houses, farms, and small businesses. Designed for realism
# with traditional security camera vantage points and enriched real-world
# conditions. Output: JSONL for Qwen-Image servers.
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

# OD classes
od_types = ["person", "animal", "vehicle", "package"]
motion_types = ["static", "non-static"]

# Traditional CCTV/security camera views (no doorbell)
camera_views = [
    "fixed_eave_corner_high",     # outdoor house/business eave, ~3–4 m
    "wall_mount_mid_height",      # outdoor/indoor wall mount, ~2–2.5 m
    "pole_mount_parking_lot",     # outdoor pole in small business lot
    "indoor_ceiling_dome",        # indoor dome camera
    "hallway_ceiling_mount",      # indoor hallway
    "warehouse_corner_mount",     # indoor warehouse corner high
    "storefront_soffit_mount",    # outdoor soffit over a shop entrance
]
camera_view_text = {
    "fixed_eave_corner_high": "fixed high-angle from building eave (~3–4 m)",
    "wall_mount_mid_height": "fixed view from wall mount (~2–2.5 m)",
    "pole_mount_parking_lot": "fixed view from parking-lot pole (~4 m)",
    "indoor_ceiling_dome": "fixed high-angle indoor dome camera (~3 m)",
    "hallway_ceiling_mount": "fixed ceiling mount down a hallway (~2.7 m)",
    "warehouse_corner_mount": "fixed high-angle from warehouse corner (~5 m)",
    "storefront_soffit_mount": "fixed soffit-mounted view over storefront (~3.5 m)",
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
    "dusk": "warm dusk light, long shadows toward camera",
    "night": "night lighting from porch or street lamps; IR night possible"
}

# Lighting conditions derived from time/property for extra realism
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
lens_effects = [None, "slight_vignetting", "minor_chromatic_aberration", "slight_barrel_distortion"]

# Colors/phrasing
colors_people = ["red jacket", "blue coat", "black hoodie", "grey clothing", "high-visibility vest", "white shirt"]
colors_vehicle = ["blue SUV", "red sedan", "white van", "grey truck", "black motorcycle"]
colors_animal = ["brown dog", "black cat", "white goat", "speckled cow", "red fox", "chicken"]
colors_package = ["brown cardboard box", "white parcel", "yellow padded mailer"]
package_places = ["on doorstep", "beside the mailbox", "on doormat", "near the front door", "under porch bench"]

# Confusers for FP traps (expanded, realistic)
confusers = {
    "person": ["mannequin", "statue", "scarecrow", "cardboard cutout", "coat rack with long coat", "shadow shaped like a person"],
    "animal": ["stuffed toy dog", "garden gnome", "animal statue", "fur coat on a chair", "blowing plastic bag"],
    "vehicle": ["toy car", "shopping cart", "wheelbarrow", "ride-on lawn mower", "dumpster"],
    "package": ["doormat", "trash bag", "shoe box lid", "folded cardboard", "plant pot", "stack of newspapers"]
}

def choose_lighting(tod, property_type, io):
    if tod == "night":
        return random.choices(
            ["porch_light_only","streetlight_sodium","floodlight_on","ir_night_monochrome","indoor_fluorescent"],
            weights=[3,2,2,3,2] if io=="outdoor" else [0,0,0,2,3])[0]
    if tod in ["dawn","dusk"]:
        return random.choice(["backlit_low_sun","overcast_diffuse","normal_daylight"])
    # day
    return random.choice(["normal_daylight","overcast_diffuse"])

def negative_prompt_base(forbid_primary=None):
    np = [
        "cartoon", "illustration", "CGI", "render", "deformed", "low-quality",
        "text", "watermark", "timestamp", "date overlay", "HUD", "CCTV overlay", "UI overlay",
        "visible camera hardware"
    ]
    if forbid_primary:
        if forbid_primary == "person":
            np.extend(["person", "people", "human", "man", "woman"])
        elif forbid_primary == "animal":
            np.extend(["animal", "dog", "cat", "cow", "goat", "fox", "chicken", "animals"])
        elif forbid_primary == "vehicle":
            np.extend(["car", "truck", "van", "motorcycle", "vehicle", "vehicles"])
        elif forbid_primary == "package":
            np.extend(["package", "parcel", "box", "delivery"])
    return ", ".join([x for x in np if x])

def background_prompt(property_type, location, io, tod, weather, lighting):
    base = []
    mapping = {
        "residential_front_porch": "residential front porch with door and steps",
        "suburban_driveway": "suburban driveway facing the street from a garage corner",
        "residential_backyard": "residential backyard with patio and lawn",
        "garage_interior": "home garage interior with shelves and parked items",
        "house_entry_hall": "house entry hallway with coat rack and mat",
        "living_room_entry": "living room entry from hallway, interior view",
        "side_gate_path": "side path along fence leading to gate",
        "farmyard_gate": "farmyard gate with fences and gravel path",
        "barn_entrance": "barn entrance with wooden doors and hay bales nearby",
        "tool_shed_doorway": "tool shed doorway with tools and equipment",
        "pasture_edge": "pasture edge with grass field and fence line",
        "tractor_parking_area": "area where farm vehicles park, gravel ground",
        "farmhouse_porch": "farmhouse porch with wooden steps and railings",
        "shop_entrance": "shop entrance with glass door and signage",
        "shop_front_counter": "shop front counter area with shelves behind",
        "backroom_storage": "backroom storage with boxes and metal racks",
        "warehouse_loading_dock": "small loading dock with roll-up door and pallets",
        "parking_lot_small": "small parking lot with painted lines",
        "office_hallway": "office hallway with doors and carpet",
        "cafe_entrance": "café entrance with awning and outdoor seating",
        "warehouse_interior_aisle": "warehouse interior aisle with tall shelving",
    }
    base.append(mapping.get(location, "generic property scene"))
    base.append("outdoor" if io == "outdoor" else "indoor")
    base.append(tod_map[tod])
    if io == "outdoor":
        base.append(wmap[weather])
    else:
        base.append("no direct weather visible")
    # property cue
    prop_text = {
        "family_house": "residential setting",
        "farm": "farm setting",
        "small_business": "small business setting"
    }[property_type]
    base.append(prop_text)
    # lighting
    if lighting == "ir_night_monochrome":
        base.append("infrared night mode, monochrome look with IR reflections")
    elif lighting == "streetlight_sodium":
        base.append("streetlight casts warm sodium-like glow")
    elif lighting == "porch_light_only":
        base.append("only porch light illuminates the area")
    elif lighting == "floodlight_on":
        base.append("bright floodlight illuminates the scene")
    elif lighting == "indoor_fluorescent":
        base.append("indoor fluorescent lighting")
    elif lighting == "warehouse_metal_halide":
        base.append("metal-halide high-bay lights")
    elif lighting == "backlit_low_sun":
        base.append("low sun backlighting")
    elif lighting == "overcast_diffuse":
        base.append("diffuse soft lighting")
    else:
        base.append("normal lighting levels")
    return ", ".join(base)

def comp_guidance(camera_view, distance, occl, extras=None):
    parts = [camera_view_text[camera_view]]
    if distance == "near": parts.append("subject fills a large part of the frame")
    if distance == "mid": parts.append("subject at mid-distance, medium shot")
    if distance == "far": parts.append("subject appears small, long shot")
    if occl >= 50:
        parts.append(f"subject {occl}% occluded by foreground object")
    elif occl >= 25:
        parts.append(f"subject partially occluded (~{occl}%)")
    if extras:
        parts.append(extras)
    return ", ".join(parts)

def pick_color_for_class(od):
    if od == "person":
        return random.choice(colors_people)
    if od == "vehicle":
        return random.choice(colors_vehicle)
    if od == "animal":
        return random.choice(colors_animal)
    if od == "package":
        return random.choice(colors_package)
    return ""

def expected_detection_dict(primary, od_state, category):
    d = {k: False for k in od_types}
    if od_state == "disabled":
        return d
    if category in ["positive", "false_negative_risk", "edge_case", "repeat_sequence"] and primary:
        d[primary] = True
    return d

# Constraint helpers ----------------------------------------------------------
def sample_property_type(od):
    weights = {"family_house": 0.5, "farm": 0.2, "small_business": 0.3}
    r = random.random(); cum = 0.0
    for pt, w in weights.items():
        cum += w
        if r <= cum:
            return pt
    return "family_house"

def sample_location_for(od, property_type):
    loc, io = random.choice(location_catalog[property_type])
    # vehicles indoor only if garage/warehouse contexts
    if od == "vehicle" and io == "indoor" and loc not in ["garage_interior", "warehouse_interior_aisle"]:
        candidates = [x for x in location_catalog[property_type] if x[1] == "outdoor"]
        loc, io = random.choice(candidates) if candidates else (loc, io)
    # packages likely at porches/entrances/hallways
    if od == "package" and loc not in ["residential_front_porch","shop_entrance","office_hallway","house_entry_hall"]:
        pref = [(cand, c_io) for cand, c_io in location_catalog[property_type]
                if cand in ["residential_front_porch","shop_entrance","office_hallway","house_entry_hall"]]
        if pref:
            loc, io = random.choice(pref)
    return loc, io

def sample_camera_view(io, property_type, location):
    if io == "indoor":
        if location in ["office_hallway","house_entry_hall","living_room_entry","shop_front_counter","backroom_storage"]:
            return "hallway_ceiling_mount" if "hallway" in location else "indoor_ceiling_dome"
        if "warehouse" in location:
            return "warehouse_corner_mount"
        return "indoor_ceiling_dome"
    # outdoor
    if location in ["parking_lot_small","warehouse_loading_dock","tractor_parking_area"]:
        return "pole_mount_parking_lot"
    if location in ["shop_entrance","cafe_entrance","residential_front_porch","farmhouse_porch"]:
        return "storefront_soffit_mount" if "shop" in location or "cafe" in location else "fixed_eave_corner_high"
    return random.choice(["fixed_eave_corner_high","wall_mount_mid_height"])

def assemble_prompt(bg, pos, comp):
    return f"photorealistic, high-detail, CCTV security camera perspective. Background: {bg}. Subject: {pos}. Composition: {comp}."

cases = []

# Generators ------------------------------------------------------------------
def gen_positive(n, start_idx):
    for k in range(n):
        i = start_idx + k
        od = random.choice(od_types)
        property_type = sample_property_type(od)
        loc, io = sample_location_for(od, property_type)
        camera_view = sample_camera_view(io, property_type, loc)
        motion = random.choice(motion_types)
        tod = random.choices(times_of_day, weights=[1,3,1,1])[0]
        weather = random.choices(weathers, weights=[5,4,2,1,1,1,1,1])[0]
        lighting = choose_lighting(tod, property_type, io)
        distance = random.choices(distances, weights=[2,3,2])[0]
        occl = random.choices(occlusion_levels, weights=[6,3,2,1,1])[0]
        lens = random.choice(lens_effects)
        artifact = random.choice(sensor_artifacts)
        color = pick_color_for_class(od)
        count = 1 if random.random() < 0.8 else random.randint(2, 4)

        bg = background_prompt(property_type, loc, io, tod, weather, lighting)
        if od == "person":
            subject = f"one {color} person {'walking' if motion=='non-static' else 'standing'}"
        elif od == "animal":
            subject = f"one {color} {'moving across scene' if motion=='non-static' else 'standing near fence'}"
        elif od == "vehicle":
            subject = f"one {color} {'driving' if motion=='non-static' else 'parked'}"
        else:
            place = random.choice(package_places)
            subject = f"one {color} placed {place}"
        if count > 1:
            subject = subject.replace("one", f"{count}").replace("person", "people")

        extras = []
        if lens and random.random() < 0.25:
            extras.append(lens.replace("_", " "))
        if artifact and random.random() < 0.3:
            extras.append(artifact.replace("_", " "))
        comp = comp_guidance(camera_view, distance, occl, ", ".join(extras) if extras else None)

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
            "negative_prompt": negative_prompt_base(),
            "composition_guidance": comp,
            "prompt": assemble_prompt(bg, subject, comp),
            "seed": random.randint(1, 2_000_000_000),
            "expected_detection": expected_detection_dict(od, "enabled", "positive"),
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
        weather = random.choice(weathers)
        lighting = choose_lighting(tod, property_type, io)
        distance = random.choice(distances)
        occl = random.choice(occlusion_levels)

        bg = background_prompt(property_type, loc, io, tod, weather, lighting)

        # 40% OD disabled => no detections expected
        state = "disabled" if random.random() < 0.4 else "enabled"

        # 50% empty, 50% other-class present
        if random.random() < 0.5:
            pos_prompt = "no primary objects, empty scene emphasis, natural background details only"
        else:
            other_od = random.choice([x for x in od_types if x != od])
            pos_prompt = f"one {pick_color_for_class(other_od)} present, but not {od}"

        comp = comp_guidance(camera_view, distance, occl)
        neg_prompt = negative_prompt_base(forbid_primary=od)

        data = {
            "test_case_id": f"OD-NEG-{i:05d}",
            "scenario_category": "negative",
            "test_subcategory": "no_detection_expected",
            "property_type": property_type,
            "od_type_primary": od,
            "od_state": state,
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
            "prompt": assemble_prompt(bg, pos_prompt, comp),
            "seed": random.randint(1, 2_000_000_000),
            "expected_detection": expected_detection_dict(None, state, "negative"),
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
        weather = random.choice(weathers)
        lighting = choose_lighting(tod, property_type, io)
        distance = random.choice(distances)
        occl = random.choice(occlusion_levels)

        conf = random.choice(confusers[od])
        bg = background_prompt(property_type, loc, io, tod, weather, lighting)
        pos_prompt = f"realistic {conf} placed naturally in scene, no actual {od} present"
        comp = comp_guidance(camera_view, distance, occl)
        neg_prompt = negative_prompt_base(forbid_primary=od)

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
            "prompt": assemble_prompt(bg, pos_prompt, comp),
            "seed": random.randint(1, 2_000_000_000),
            "expected_detection": expected_detection_dict(None, "enabled", "false_positive_trap"),
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
        weather = random.choice(["fog_dense","rain_heavy","snow_heavy"] if random.random()<0.7 else weathers)
        lighting = choose_lighting(tod, property_type, io)
        distance = random.choice(["far","mid"])
        occl = random.choice([50,75,90])
        artifact = random.choice(["motion_blur","low_resolution","image_noise","rolling_shutter_skew"])

        bg = background_prompt(property_type, loc, io, tod, weather, lighting)
        color = pick_color_for_class(od)
        if od == "person":
            subject = f"one {color} partially hidden behind a fence"
        elif od == "animal":
            subject = f"one {color} partly behind bushes"
        elif od == "vehicle":
            subject = f"one {color} mostly hidden behind another parked vehicle"
        else:
            subject = f"one {color} tucked in a shaded corner on the ground"
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
            "negative_prompt": negative_prompt_base(),
            "composition_guidance": comp,
            "prompt": assemble_prompt(bg, subject, comp),
            "seed": random.randint(1, 2_000_000_000),
            "expected_detection": expected_detection_dict(od, "enabled", "false_negative_risk"),
            "risk_tags": ["false_negative_risk", "hard_conditions"]
        }
        cases.append(data)

def gen_edge_cases(n, start_idx):
    edge_subs = [
        ("extreme_backlight","strong backlight, silhouette tendency, lens flare"),
        ("lens_obstruction","raindrops, dirt smears, or spider webs on the lens"),
        ("headlights_glare","vehicle headlights directed at camera, glare on wet pavement"),
        ("corner_entry","subject entering from extreme corner"),
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
        weather = random.choice(weathers)
        lighting = choose_lighting(tod, property_type, io)
        distance = random.choice(distances)
        occl = random.choice(occlusion_levels)
        sub = random.choice(edge_subs)

        bg = background_prompt(property_type, loc, io, tod, weather, lighting)
        color = pick_color_for_class(od)
        subject_map = {
            "person": f"one {color} {'moving' if motion=='non-static' else 'standing'}",
            "animal": f"one {color} {'running' if motion=='non-static' else 'standing'}",
            "vehicle": f"one {color} {'moving' if motion=='non-static' else 'parked'}",
            "package": f"one {color} on ground"
        }
        subject = subject_map[od]
        comp = comp_guidance(camera_view, distance, occl, extras=sub[1])
        if sub[0] == "crowded_scene" and od == "person":
            subject = subject.replace("one", str(random.randint(5, 20))).replace("person", "people")

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
            "negative_prompt": negative_prompt_base(),
            "composition_guidance": comp,
            "prompt": assemble_prompt(bg, subject, comp),
            "seed": random.randint(1, 2_000_000_000),
            "expected_detection": expected_detection_dict(od, "enabled", "edge_case"),
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
        weather = random.choice(weathers)
        lighting = choose_lighting(tod, property_type, io)
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
            occl = min(90, occl_base + f*5 if pattern in ["enter_stop_exit","enter_exit_reenter","parking_lot_loops"] else occl_base)
            motion = "non-static" if pattern not in ["enter_stop_exit","package_drop_and_removal"] or f in [0,4] else ("static" if f in [2] else "non-static")
            bg = background_prompt(property_type, loc, io, tod, weather, lighting)

            color = pick_color_for_class(od)
            if od == "person":
                subject = f"one {color} {'walking' if motion=='non-static' else 'standing'}"
                if pattern == "linger_by_door" and f in [1,2,3]:
                    subject += ", lingering near the doorway"
            elif od == "animal":
                subject = f"one {color} {'moving' if motion=='non-static' else 'still'}"
                if pattern == "animal_in_and_out":
                    subject += ", repeatedly entering and leaving frame"
            elif od == "vehicle":
                subject = f"one {color} {'moving across frame' if motion=='non-static' else 'stopped briefly'}"
                if pattern == "parking_lot_loops":
                    subject += ", looping across parking lot"
            else:
                subject = f"one {color} {'newly placed' if f==1 else ('still there' if f in [2,3] else 'removed')} on the ground"

            comp = comp_guidance(camera_view, distance, occl)
            if pattern == "enter_exit_reenter" and f == 3:
                comp += ", object re-enters frame"
            if pattern == "multi_passes":
                comp += ", object passes across frame multiple times"
            if pattern == "package_drop_and_removal":
                od = "package"  # force package sequence

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
                "negative_prompt": negative_prompt_base(),
                "composition_guidance": comp,
                "prompt": assemble_prompt(bg, subject, comp),
                "seed": random.randint(1, 2_000_000_000),
                "expected_detection": expected_detection_dict(od, "enabled", "repeat_sequence"),
                "risk_tags": ["repeat_duplicate_notifications"]
            }
            cases.append(data)
            frame_i += 1

# Generate according to distribution
idx = 1
gen_positive(DISTRIBUTION["positive"], idx); idx += DISTRIBUTION["positive"]
gen_negative(DISTRIBUTION["negative"], idx); idx += DISTRIBUTION["negative"]
gen_false_positive_trap(DISTRIBUTION["false_positive_trap"], idx); idx += DISTRIBUTION["false_positive_trap"]
gen_false_negative_risk(DISTRIBUTION["false_negative_risk"], idx); idx += DISTRIBUTION["false_negative_risk"]
gen_edge_cases(DISTRIBUTION["edge_case"], idx); idx += DISTRIBUTION["edge_case"]
gen_repeat_sequences(DISTRIBUTION["repeat_sequence"], idx); idx += DISTRIBUTION["repeat_sequence"]

# Shuffle for variety
random.shuffle(cases)

# Write JSONL
out_path = Path("../dataset/od_synth_cases_10000_cctv.jsonl")
with out_path.open("w", encoding="utf-8") as f:
    for row in cases:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# out_path.as_posix()
