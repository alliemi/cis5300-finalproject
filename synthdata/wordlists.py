"""Fixed word lists and sentence templates for the synthetic retrieval data.

Everything the generator can ever write comes from this file, so the
vocabulary is closed. Three groups of words are kept strictly apart
(checked by `check_wordlists` and by the tests):

* NAMES   - capitalized project names ("Falcon"); one token each
* VALUES  - lowercase status values ("shipped"); one token each
* filler  - words used in non-fact sentences; never a name or a value

Because names and values never appear outside fact sentences, any value a
model predicts can be traced to exactly one fact (or to "not in context").
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Fact and query templates. "{E}" = entity (project name), "{V}" = value.
# Templates are token tuples, so their lengths are known exactly (6, 5, 8).
# ---------------------------------------------------------------------------
FACT_TEMPLATES = (
    ("Project", "{E}", "status", ":", "{V}", "."),
    ("Project", "{E}", "is", "{V}", "."),
    ("The", "status", "of", "Project", "{E}", "is", "{V}", "."),
)
QUERY_TEMPLATE = ("What", "is", "the", "current", "status", "of", "Project", "{E}", "?")

# Words reserved for facts and queries; filler may never use these.
RESERVED_WORDS = ("Project", "status")

# ---------------------------------------------------------------------------
# Project names (capitalized, letters only).
# ---------------------------------------------------------------------------
_NAMES = """
Falcon Heron Otter Badger Beaver Bison Bobcat Buffalo Camel Caribou Cheetah
Cobra Condor Cougar Coyote Crane Crow Dingo Dolphin Donkey Dove Eagle Egret Elk
Emu Ferret Finch Fox Gazelle Gecko Gibbon Giraffe Gopher Gorilla Grouse Gull
Hamster Hare Hawk Hedgehog Hippo Hornet Horse Hyena Ibis Iguana Impala Jackal
Jaguar Jay Kestrel Kingfisher Kiwi Koala Lemur Leopard Lion Lizard Llama
Lobster Lynx Macaw Magpie Mallard Mamba Manatee Mantis Marlin Marmot Meerkat
Mink Mole Mongoose Moose Moth Mule Narwhal Newt Ocelot Octopus Opossum Orca
Oriole Osprey Ostrich Owl Panda Panther Parrot Pelican Penguin Pheasant Pigeon
Pike Plover Puffin Puma Python Quail Rabbit Raccoon Raven Robin Seal Shark
Shrew Skunk Sloth Snail Sparrow Spider Squid Stork Swallow Swan Tapir Tern
Tiger Toad Toucan Trout Tuna Turkey Turtle Viper Vulture Walrus Wasp Weasel
Whale Wolf Wombat Wren Yak Zebra
Acacia Alder Aspen Birch Cypress Elm Fir Hazel Hemlock Juniper Larch Laurel
Linden Magnolia Oak Palm Pine Poplar Redwood Rowan Sequoia Spruce Sycamore
Willow Yew
Altair Antares Arcturus Betelgeuse Capella Castor Deneb Polaris Procyon Rigel
Sirius Spica Vega Pollux Mira Orion Lyra Draco Cygnus Hydra Pegasus Perseus
Phoenix Andromeda Cassiopeia Aquila Carina Centaurus Corvus Lupus Vela
Everest Denali Rainier Shasta Whitney Kilimanjaro Elbrus Olympus Etna Fuji
Vesuvius Amazon Nile Danube Rhine Volga Yukon Congo Mekong Ganges Thames Seine
Hudson Columbia Tigris Indus
Argon Boron Neon Xenon Krypton Radon Helium Lithium Titan Atlas Apollo Gemini
Mercury Saturn Jupiter Neptune Pluto
Alpha Beta Gamma Delta Epsilon Zeta Theta Iota Kappa Lambda Sigma Omega
Omicron Upsilon
Athena Hermes Zeus Hera Thor Odin Loki Freya Juno Vulcan Janus Minerva
"""

# ---------------------------------------------------------------------------
# Status values (lowercase, letters only). Many are not literal statuses;
# that is fine for a model trained from scratch and gives a large pool.
# ---------------------------------------------------------------------------
_VALUES = """
shipped blocked testing paused delayed cancelled approved archived drafted
deployed released merged stalled reviewed planned scoped funded frozen pending
started finished rejected launched ready staged queued audited deferred halted
resumed signed closed active idle stable broken degraded
red blue green amber violet indigo crimson scarlet teal cyan magenta maroon
navy olive ivory beige coral khaki lilac mauve ochre plum rust sepia tan
turquoise azure emerald jade ruby sapphire topaz garnet opal pearl onyx quartz
amethyst
copper bronze silver golden iron steel tin zinc nickel chrome marble granite
slate clay brick glass cotton linen silk wool velvet denim leather bamboo
banana cherry grape lemon lime mango melon peach pear fig papaya guava lychee
apricot cocoa vanilla ginger pepper cinnamon nutmeg saffron mint basil sage
thyme fennel clove honey caramel toffee
sunny rainy cloudy windy foggy snowy stormy misty frosty humid breezy calm
brisk bright dim dusty fuzzy glossy hollow jagged lumpy matte noisy rapid rigid
silent sleek smooth soggy solid spiky sticky tidy vivid wavy woolly
tango waltz polka salsa samba rumba mambo bolero jazz blues funk disco opera
ballad anthem sonata
circle square oval cube sphere prism spiral zigzag
anchor arrow beacon candle compass feather harbor lantern meadow orchard pebble
prairie quiver ribbon saddle tundra valley canyon glacier lagoon marsh oasis
summit
walnut almond cashew pistachio chestnut acorn radish carrot celery turnip potato
tomato onion garlic pumpkin squash cabbage spinach kale lettuce cucumber parsley
dill oregano rosemary
crystal diamond flint basalt pumice obsidian chalk
thistle clover daisy tulip lily orchid poppy lotus iris jasmine lavender
cello flute harp lute oboe tuba banjo fiddle bugle piano
"""

NAMES = tuple(sorted(set(_NAMES.split())))
VALUES = tuple(sorted(set(_VALUES.split())))

# ---------------------------------------------------------------------------
# Filler grammar. A filler sentence is one of:
#   [opener] The <subject> <predicate> [time] .
#   [opener] Someone left a <item> in the <place> .
#   [opener] Please remember to <action> .
# After an opener, "The" becomes "the".
# ---------------------------------------------------------------------------
FILLER_OPENERS = ("", "Meanwhile ,", "Also ,", "In other news ,", "As usual ,", "Earlier ,", "Reminder :")
FILLER_SUBJECTS = (
    "team lunch", "weekly meeting", "office printer", "coffee machine",
    "parking garage", "front desk", "new intern", "fire drill", "kitchen fridge",
    "conference room", "holiday party", "badge reader", "office wifi",
    "budget report", "hiring committee", "design meeting", "town hall",
    "shuttle bus", "break room", "vending machine", "elevator", "lobby plant",
    "supply closet", "phone line", "training session", "security desk",
    "mail room", "air conditioner", "snack drawer", "board meeting",
    "visitor log", "travel policy", "dress code", "desk lamp", "water cooler",
    "cafeteria menu", "lost and found box",
)
FILLER_PREDICATES = (
    "was moved", "ran late", "needs repair", "is being cleaned",
    "was rescheduled", "got a new sign", "will be shut", "had a long line",
    "is on the third floor", "was moved upstairs", "is out of paper",
    "made a strange sound", "was painted", "got more chairs",
    "was mentioned twice", "needs a new filter", "was replaced",
    "is fully booked", "was updated", "starts at nine", "ends at five",
    "has a new manager", "was discussed at length", "needs more volunteers",
    "was a big hit", "is still running", "smells like popcorn",
    "was left unlocked", "has a note on it", "was very crowded",
)
FILLER_TIMES = (
    "", "on Monday", "on Tuesday", "on Wednesday", "on Thursday", "on Friday",
    "this week", "next week", "after lunch", "in the morning", "before noon",
    "again today", "last night", "this afternoon",
)
FILLER_ITEMS = ("mug", "jacket", "umbrella", "notebook", "charger", "scarf", "lunch bag", "water bottle")
FILLER_PLACES = ("kitchen", "lobby", "break room", "elevator", "mail room", "conference room")
FILLER_ACTIONS = (
    "sign the visitor log", "refill the water cooler", "clean the kitchen",
    "lock the supply closet", "turn off the lights", "submit your timesheet",
    "water the lobby plant", "badge in at the front desk",
)
# Short sentences of 2, 3 and 4 tokens, used to land exactly on a length.
FILLER_PADS = {
    2: ("Noted .", "Agreed .", "Thanks ."),
    3: ("More soon .", "Thanks everyone .", "Okay then ."),
    4: ("Nothing else changed .", "That is all .", "See you later ."),
}

_WORD_RE = re.compile(r"^[A-Za-z]+$")


def filler_phrases():
    """Every phrase string the filler grammar can use (for vocab building)."""
    out = list(FILLER_OPENERS) + list(FILLER_SUBJECTS) + list(FILLER_PREDICATES)
    out += list(FILLER_TIMES) + list(FILLER_ITEMS) + list(FILLER_PLACES)
    out += list(FILLER_ACTIONS)
    out += ["The", "the", "Someone left a", "someone", "in the", "Please remember to", "please", "."]
    for pads in FILLER_PADS.values():
        out += list(pads)
    return [p for p in out if p]


def filler_words():
    return {w for p in filler_phrases() for w in p.split()}


def check_wordlists():
    """Raise AssertionError if the word groups overlap or are malformed."""
    assert len(NAMES) >= 256, len(NAMES)
    assert len(VALUES) >= 256, len(VALUES)
    for n in NAMES:
        assert _WORD_RE.match(n) and n[0].isupper() and n[1:].islower(), n
    for v in VALUES:
        assert _WORD_RE.match(v) and v.islower(), v
    names_lc = {n.lower() for n in NAMES}
    values_lc = {v.lower() for v in VALUES}
    filler_lc = {w.lower() for w in filler_words()}
    template_lc = {w.lower() for t in FACT_TEMPLATES + (QUERY_TEMPLATE,) for w in t if "{" not in w}
    assert not names_lc & values_lc, sorted(names_lc & values_lc)
    assert not names_lc & filler_lc, sorted(names_lc & filler_lc)
    assert not values_lc & filler_lc, sorted(values_lc & filler_lc)
    assert not (names_lc | values_lc) & template_lc, sorted((names_lc | values_lc) & template_lc)
    reserved_lc = {w.lower() for w in RESERVED_WORDS}
    assert not reserved_lc & filler_lc, sorted(reserved_lc & filler_lc)
    for n, pads in FILLER_PADS.items():
        for p in pads:
            assert len(p.split()) == n, p
