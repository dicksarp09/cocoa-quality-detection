
translation_map = {
    "good": "Good quality cocoa bean",
    "poor": "Poor quality cocoa bean"
}

def translate_class(label):
    return translation_map.get(label, f"Unknown class: {label}")
