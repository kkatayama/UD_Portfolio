SELECT name,originalType,originalText,flavorText,number,setCode,artist, MATCH (name, originalType, originalText, flavorText, number, setCode, artist) AGAINST ('Anje Falkenrath Legendary Creature — Vampire Discard a card: Draw a card. Whenever you discard a card, if it has madness, untap Anje Falkenrath. “We all hide a little madness behind our sophistication, do we not?” 037/302 M C19 EN CYNTHIA SHEPPARD & © 2019 Wizards of the Coast') AS score FROM cards WHERE MATCH (name, originalType, originalText, flavorText, number, setCode, artist) AGAINST ('Anje Falkenrath Legendary Creature — Vampire Discard a card: Draw a card. Whenever you discard a card, if it has madness, untap Anje Falkenrath. “We all hide a little madness behind our sophistication, do we not?” 037/302 M C19 EN CYNTHIA SHEPPARD & © 2019 Wizards of the Coast') LIMIT 3 \G
