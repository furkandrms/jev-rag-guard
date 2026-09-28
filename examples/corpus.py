"""A small, original corpus for `real_pipeline.py`.

Short, factual, hand-written blurbs (not scraped or reproduced from any
copyrighted source) covering a handful of unrelated topics, so relevance
filtering and sufficiency gating have real off-topic noise to reject.
"""

from __future__ import annotations

DOCUMENTS: list[tuple[str, str]] = [
    ("geo1", "Paris is the capital and most populous city of France."),
    ("geo2", "France is a country in Western Europe with a population of about 68 million."),
    ("geo3", "The Eiffel Tower is a wrought-iron lattice tower in Paris, completed in 1889."),
    ("geo4", "The Seine river flows through Paris before emptying into the English Channel."),
    ("geo5", "Tokyo is the capital of Japan and one of the most populous metropolitan areas in the world."),
    ("geo6", "Japan is an island country in East Asia made up of four main islands."),
    ("geo7", "Mount Fuji, Japan's tallest mountain, is an active stratovolcano near Tokyo."),
    ("geo8", "Canberra, not Sydney, is the capital city of Australia."),
    ("geo9", "Australia is both a country and a continent, located in the Southern Hemisphere."),
    ("geo10", "Brasilia was purpose-built in the 1960s to become Brazil's capital."),
    ("sci1", "Water boils at 100 degrees Celsius at standard atmospheric pressure."),
    ("sci2", "Photosynthesis is the process plants use to convert sunlight into chemical energy."),
    ("sci3", "The speed of light in a vacuum is approximately 299,792 kilometers per second."),
    ("sci4", "DNA carries the genetic instructions used in the growth and development of organisms."),
    ("sci5", "Jupiter is the largest planet in the Solar System and has at least 95 known moons."),
    ("sci6", "The human body has 206 bones in adulthood, down from about 270 at birth."),
    ("sci7", "Electrons carry a negative electric charge and orbit an atom's nucleus."),
    ("sci8", "A leap year occurs roughly every four years to keep the calendar aligned with Earth's orbit."),
    ("food1", "Bananas are a good source of potassium and are grown in tropical climates."),
    ("food2", "Espresso is brewed by forcing hot water under pressure through finely-ground coffee."),
    ("food3", "Sourdough bread rises using a fermented culture of wild yeast and bacteria."),
    ("food4", "Honey never spoils if stored properly, thanks to its low moisture content and acidity."),
    ("hist1", "The Roman Empire at its height stretched from Britain to the Middle East."),
    ("hist2", "Gutenberg's printing press, developed around 1440, transformed how information spread."),
    ("hist3", "The Apollo 11 mission landed the first humans on the Moon in July 1969."),
    ("hist4", "The Berlin Wall fell in November 1989, symbolizing the end of Europe's Cold War divide."),
    ("tech1", "Python is a general-purpose programming language known for its readable syntax."),
    ("tech2", "A vector database stores embeddings and supports approximate nearest-neighbor search."),
    ("tech3", "HTTP is a stateless protocol used for transferring data on the World Wide Web."),
    ("tech4", "A hash function maps input data of arbitrary size to a fixed-size output."),
]
