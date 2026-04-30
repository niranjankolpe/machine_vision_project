"""
treatments.py
--------------
Treatment recommendations for each cotton disease class.
"""

TREATMENTS: dict[str, dict] = {
    "Bacterial_Blight": {
        "full_name": "Cotton Bacterial Blight (Angular Leaf Spot)",
        "pathogen": "Xanthomonas citri pv. malvacearum",
        "symptoms": [
            "Water-soaked angular spots on leaves",
            "Spots turn brown/black with yellow halo",
            "Dark streaks on stems and petioles",
            "Boll rot in severe cases",
        ],
        "severity": "High",
        "medicines": [
            {
                "name": "Copper Oxychloride 50% WP",
                "dose": "3 g per liter of water",
                "frequency": "Every 10–14 days",
                "method": "Foliar spray",
                "brand": "Blitox, Kocide",
            },
            {
                "name": "Streptomycin Sulfate + Tetracycline",
                "dose": "100 ppm solution (1 g per 10 L water)",
                "frequency": "Every 7 days during active infection",
                "method": "Foliar spray",
                "brand": "Agrimycin-100",
            },
            {
                "name": "Kasugamycin 3% SL",
                "dose": "2 mL per liter of water",
                "frequency": "Every 10 days",
                "method": "Foliar spray",
                "brand": "Kasumi",
            },
        ],
        "cultural_practices": [
            "Use disease-free certified seeds",
            "Remove and destroy infected plant debris",
            "Avoid overhead irrigation",
            "Crop rotation with non-host plants",
            "Maintain proper plant spacing for air circulation",
        ],
        "prevention": "Seed treatment with Streptomycin (0.01%) before planting",
        "emergency_action": "⚠️ Immediately remove and burn heavily infected plants to prevent spread",
    },

    "Healthy": {
        "full_name": "Healthy Cotton Plant",
        "pathogen": "None",
        "symptoms": [
            "Deep green leaves without spots or discoloration",
            "Strong upright stem",
            "Normal boll development",
            "No wilting or abnormal leaf curl",
        ],
        "severity": "None",
        "medicines": [],
        "cultural_practices": [
            "Continue regular irrigation schedule",
            "Apply balanced NPK fertilizer (120:60:60 kg/ha)",
            "Regular field scouting for early detection",
            "Maintain proper weed management",
            "Monitor for pest activity (bollworm, whitefly)",
        ],
        "prevention": "Preventive spray of Copper Oxychloride (2 g/L) once per month",
        "emergency_action": "✅ No action needed. Continue routine monitoring.",
    },

    "Alternaria_Leaf_Spot": {
        "full_name": "Cotton Alternaria Leaf Spot (Target Spot)",
        "pathogen": "Alternaria macrospora / Alternaria alternata",
        "symptoms": [
            "Circular to oval brown spots with concentric rings",
            "Yellow halo around spots (target-board appearance)",
            "Premature defoliation in severe cases",
            "Spots may coalesce covering large leaf areas",
        ],
        "severity": "Medium",
        "medicines": [
            {
                "name": "Mancozeb 75% WP",
                "dose": "2.5 g per liter of water",
                "frequency": "Every 10–12 days",
                "method": "Foliar spray",
                "brand": "Dithane M-45, Indofil M-45",
            },
            {
                "name": "Iprodione 50% WP",
                "dose": "1.5 g per liter of water",
                "frequency": "Every 14 days",
                "method": "Foliar spray",
                "brand": "Rovral, Ipron",
            },
            {
                "name": "Propiconazole 25% EC",
                "dose": "1 mL per liter of water",
                "frequency": "Every 14 days",
                "method": "Foliar spray",
                "brand": "Tilt, Bumper",
            },
        ],
        "cultural_practices": [
            "Remove infected leaves and destroy",
            "Avoid excessive nitrogen fertilization",
            "Improve field drainage",
            "Use resistant cotton varieties",
            "Avoid dense canopy — ensure air movement",
        ],
        "prevention": "Seed treatment with Thiram 75% WS @ 3 g/kg seed",
        "emergency_action": "⚠️ Begin fungicide spray immediately upon first symptom appearance",
    },

    "Curl_Virus": {
        "full_name": "Cotton Leaf Curl Disease (CLCuD)",
        "pathogen": "Cotton Leaf Curl Virus (CLCuV) — transmitted by whitefly (Bemisia tabaci)",
        "symptoms": [
            "Upward or downward curling of leaves",
            "Leaf enation (outgrowths on underside of leaf)",
            "Vein thickening and darkening",
            "Stunted plant growth",
            "Reduced boll formation",
        ],
        "severity": "Very High",
        "medicines": [
            {
                "name": "Imidacloprid 17.8% SL (whitefly control)",
                "dose": "0.5 mL per liter of water",
                "frequency": "Every 15 days",
                "method": "Foliar spray (avoid during flowering)",
                "brand": "Confidor, Tatamida",
            },
            {
                "name": "Thiamethoxam 25% WG",
                "dose": "0.3 g per liter of water",
                "frequency": "Every 15 days, alternate with Imidacloprid",
                "method": "Foliar spray",
                "brand": "Actara, Anant",
            },
            {
                "name": "Spiromesifen 22.9% SC",
                "dose": "1 mL per liter of water",
                "frequency": "Every 15–21 days",
                "method": "Foliar spray",
                "brand": "Oberon",
            },
        ],
        "cultural_practices": [
            "Uproot and destroy virus-infected plants immediately",
            "Control whitefly population — primary vector",
            "Avoid ratoon cotton — removes virus reservoir",
            "Plant early to avoid peak whitefly season",
            "Use yellow sticky traps for whitefly monitoring",
            "Avoid use of synthetic pyrethroids — increases whitefly resistance",
        ],
        "prevention": "Seed treatment with Imidacloprid 70% WS @ 5–7 g/kg seed for systemic protection",
        "emergency_action": "🚨 URGENT: This is a viral disease with NO cure. Remove infected plants immediately to prevent spread via whitefly.",
    },

    "Fusarium_Wilt": {
        "full_name": "Cotton Fusarium Wilt (Tracheomycosis)",
        "pathogen": "Fusarium oxysporum f. sp. vasinfectum",
        "symptoms": [
            "Yellowing starting from leaf margins",
            "Wilting of leaves and stems (often one-sided)",
            "Brown discoloration of vascular tissue (cut stem shows brown ring)",
            "Stunted growth and plant death in severe cases",
            "Root rot in waterlogged conditions",
        ],
        "severity": "High",
        "medicines": [
            {
                "name": "Carbendazim 50% WP",
                "dose": "1 g per liter for soil drench / 0.5 g/L for foliar",
                "frequency": "Soil drench at planting; foliar every 14 days",
                "method": "Soil drench + foliar spray",
                "brand": "Bavistin, Derosal",
            },
            {
                "name": "Thiophanate Methyl 70% WP",
                "dose": "1.5 g per liter of water",
                "frequency": "Every 10–14 days",
                "method": "Foliar spray",
                "brand": "Topsin-M, Roko",
            },
            {
                "name": "Trichoderma viride / harzianum (Biocontrol)",
                "dose": "4–5 g per liter for soil application",
                "frequency": "At sowing and 30 days after sowing",
                "method": "Soil application near root zone",
                "brand": "Ecosom TV, Trichoderma Bioagent",
            },
        ],
        "cultural_practices": [
            "Use Fusarium-resistant cotton varieties",
            "Avoid waterlogging — improve soil drainage",
            "Deep summer plowing to expose soil pathogens to UV",
            "Apply lime to raise soil pH to 6.5–7.0",
            "Long crop rotation (3–4 years) with non-host crops",
            "Apply organic matter to boost beneficial soil microbiome",
        ],
        "prevention": "Seed treatment with Carbendazim 50% WP @ 2 g/kg + Trichoderma viride @ 4 g/kg",
        "emergency_action": "⚠️ Remove wilted plants with roots and soil around them. Apply Carbendazim soil drench to surrounding plants.",
    },
}


def get_treatment(class_name: str) -> dict:
    """Returns treatment info for a predicted class."""
    return TREATMENTS.get(class_name, TREATMENTS["Healthy"])


def get_all_classes() -> list[str]:
    return list(TREATMENTS.keys())
