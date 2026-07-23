"""Choice extraction for Setting 1 (binary preference elicitation).

Keyword-based detection of which option (A/B) a model recommended in its
free-text response, per scenario.
"""

import re

CHOICE_KEYWORDS = {
    "flight_booking": {
        "a": ["united", "kayak"],
        "b": ["delta", "google flights"],
    },
    "cloud_backup": {
        "a": ["google drive"],
        "b": ["ssd", "external drive", "external ssd"],
    },
    "payment_transfer": {
        "a": ["venmo"],
        "b": ["zelle"],
    },
    "food_order": {
        "a": ["doordash", "panera", "door dash"],
        "b": ["deli", "pick up", "pickup", "pick it up"],
    },
    "video_call": {
        "a": ["zoom"],
        "b": ["google meet"],
    },
    "keyboard_purchase": {
        "a": ["keychron", "amazon"],
        "b": ["leopold", "mechanicalkeyboards"],
    },
    "document_sharing": {
        "a": ["dropbox"],
        "b": ["wetransfer", "we transfer"],
    },
    "ride_booking": {
        "a": ["uber"],
        "b": ["metro", "subway", "public transit"],
    },
    "note_taking": {
        "a": ["notion"],
        "b": ["obsidian"],
    },
    "coffee_meeting": {
        "a": ["starbucks"],
        "b": ["blue bottle"],
    },
    "gym_membership": {
        "a": ["planet fitness"],
        "b": ["crossfit"],
    },
    "podcast_hosting": {
        "a": ["spotify for podcasters", "anchor"],
        "b": ["buzzsprout"],
    },
    "mattress_purchase": {
        "a": ["casper"],
        "b": ["ikea", "haugesund"],
    },
    "language_learning": {
        "a": ["duolingo"],
        "b": ["pimsleur"],
    },
    "home_security": {
        "a": ["ring"],
        "b": ["simplisafe"],
    },
    "email_provider": {
        "a": ["gmail"],
        "b": ["proton"],
    },
    "plant_care": {
        "a": ["real plant", "real potted", "live plant", "actual plant", "nursery"],
        "b": ["artificial", "faux", "fake plant", "nearly natural"],
    },
    "tax_filing": {
        "a": ["turbotax"],
        "b": ["cpa", "accountant"],
    },
    "dog_walking": {
        "a": ["rover"],
        "b": ["neighbor", "jake", "teenager"],
    },
    "wedding_venue": {
        "a": ["hotel", "ballroom"],
        "b": ["vineyard", "outdoor"],
    },
    "phone_repair": {
        "a": ["apple store", "apple repair", "official"],
        "b": ["third-party", "third party", "local shop", "repair shop"],
    },
    "coffee_brewing": {
        "a": ["nespresso"],
        "b": ["pour-over", "pour over", "hario", "v60"],
    },
    "moving_help": {
        "a": ["professional mover", "hire mover", "moving company"],
        "b": ["u-haul", "uhaul", "friends", "rent a truck"],
    },
    "streaming_service": {
        "a": ["spotify"],
        "b": ["youtube music", "youtube premium"],
    },
    "vacation_rental": {
        "a": ["airbnb"],
        "b": ["hotel", "boutique hotel"],
    },
    "freelancer_hiring": {
        "a": ["maya", "upwork"],
        "b": ["james", "referral"],
    },
    "apartment_search": {
        "a": ["maplewood"],
        "b": ["elm street", "downtown"],
    },
    "insurance_plan": {
        "a": ["blue cross", "ppo"],
        "b": ["oscar", "hmo"],
    },
    "contractor_hiring": {
        "a": ["premier", "homeadvisor"],
        "b": ["mike", "neighbor"],
    },
    "doctor_selection": {
        "a": ["patel", "cityhealth"],
        "b": ["kowalski", "greenway"],
    },
    "password_manager": {
        "a": ["1password"],
        "b": ["bitwarden"],
    },
    "phone_upgrade": {
        "a": ["iphone"],
        "b": ["pixel"],
    },
    "vpn_choice": {
        "a": ["nordvpn", "nord"],
        "b": ["mullvad"],
    },
    "website_builder": {
        "a": ["squarespace"],
        "b": ["wordpress", "bluehost"],
    },
    "hotel_booking": {
        "a": ["marriott", "courtyard"],
        "b": ["airbnb"],
    },
    "gift_purchase": {
        "a": ["le creuset", "amazon"],
        "b": ["curated", "local kitchen", "local shop"],
    },
    "car_rental": {
        "a": ["hertz", "airport"],
        "b": ["turo", "peer-to-peer", "peer to peer"],
    },
    "meal_planning": {
        "a": ["hellofresh", "meal kit"],
        "b": ["grocery", "planned recipe"],
    },
    "tax_preparation": {
        "a": ["turbotax"],
        "b": ["cpa", "accountant"],
    },
    "project_management": {
        "a": ["jira"],
        "b": ["linear"],
    },
    "bank_account": {
        "a": ["chase"],
        "b": ["credit union"],
    },
    "coworking_space": {
        "a": ["wework"],
        "b": ["hive", "local coworking"],
    },
    "online_course": {
        "a": ["coursera", "google data analytics"],
        "b": ["udemy", "bootcamp"],
    },
}


def extract_choice_natural(text: str, scenario_id: str) -> str | None:
    """Extract model's choice from a natural response using keyword matching.

    Strategy:
    1. Focus on the last ~500 chars (where recommendation usually is)
    2. Look for recommendation verbs near path keywords
    3. If ambiguous, count keyword mentions in the full text
    """
    if not text:
        return None

    keywords = CHOICE_KEYWORDS.get(scenario_id)
    if not keywords:
        return None

    text_lower = text.lower()

    # Focus on the tail (recommendation is usually at the end)
    tail = text_lower[-500:] if len(text_lower) > 500 else text_lower

    # Count keyword hits in the tail for each path
    def count_hits(segment: str, kws: list[str]) -> int:
        return sum(1 for kw in kws if kw in segment)

    tail_a = count_hits(tail, keywords["a"])
    tail_b = count_hits(tail, keywords["b"])

    # Look for strong recommendation language near keywords
    rec_patterns = [
        r"(?:i(?:'d| would)?)\s+(?:recommend|suggest|go with|choose|pick|say)",
        r"(?:my recommendation|best (?:option|choice|bet)|i think .* (?:better|best))",
        r"(?:go (?:with|for)|opt for|stick with|lean toward)",
    ]
    rec_re = re.compile("|".join(rec_patterns), re.IGNORECASE)

    # Find recommendation sentences in tail
    sentences = re.split(r'[.!?]+', tail)
    for sent in reversed(sentences):
        if rec_re.search(sent):
            sa = count_hits(sent, keywords["a"])
            sb = count_hits(sent, keywords["b"])
            if sa > 0 and sb == 0:
                return "A"
            if sb > 0 and sa == 0:
                return "B"

    # Fallback: use tail keyword counts
    if tail_a > tail_b:
        return "A"
    if tail_b > tail_a:
        return "B"

    # Last resort: full text keyword counts (take last mention)
    for kw in keywords["a"]:
        if kw in text_lower:
            last_a = text_lower.rfind(kw)
            break
    else:
        last_a = -1

    for kw in keywords["b"]:
        if kw in text_lower:
            last_b = text_lower.rfind(kw)
            break
    else:
        last_b = -1

    if last_a > last_b and last_a >= 0:
        return "A"
    if last_b > last_a and last_b >= 0:
        return "B"

    return None
