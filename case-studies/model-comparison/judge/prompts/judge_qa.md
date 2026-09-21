Beoordeel een Nederlandstalig vraag-antwoordpaar, dat automatisch is
gegenereerd op basis van het Wikipedia-artikel hieronder. Dit paar is bedoeld
als trainingsdata voor een taalmodel; jouw beoordeling bepaalt of het
bruikbaar is afhankelijk van de kwaliteit.

Geef elk criterium een geheel getal van 1 tot en met 5. Gebruik de volledige
schaal: 3 is het normale oordeel voor "grotendeels in orde, maar met een
duidelijk mankement", en 5 is voorbehouden aan werk waar niets op aan te
merken valt. Beoordeel elk criterium apart; laat een zwakke score op het ene
criterium niet doorwegen op het andere.

- **question_relevant**: gaat de vraag daadwerkelijk over de inhoud van het
  artikel, en peilt ze naar een concreet feit, gebeurtenis, definitie of
  verband?
  1 = gaat niet over het artikel, of vraagt naar iets volstrekt triviaals;
  3 = gaat over het artikel, maar naar een randdetail of erg vaag geformuleerd;
  5 = peilt naar een concreet, inhoudelijk kernpunt van het artikel.
- **question_self_contained**: staat de vraag op zichzelf? Iemand die het
  artikel nooit gezien heeft, moet ze kunnen lezen en begrijpen waar ze over
  gaat.
  1 = verwijst expliciet naar "dit artikel", "de tekst hierboven",
  "bovenstaande" of iets van gelijke strekking, en is zonder het artikel
  betekenisloos;
  3 = geen expliciete verwijzing, maar mist context (bv. een "hij" of "het
  bedrijf" zonder dat duidelijk is wie of wat bedoeld wordt);
  5 = volledig zelfstandig leesbaar, met alle nodige context in de vraag zelf.
- **answer_correct**: is het antwoord feitelijk juist volgens het artikel?
  1 = spreekt het artikel tegen, of is gewoon fout;
  3 = de kern klopt, maar een detail (getal, datum, naam) is fout of
  onnauwkeurig;
  5 = volledig juist volgens het artikel.
- **answer_grounded**: steunt het antwoord uitsluitend op wat er letterlijk of
  logisch afleidbaar in het artikel staat, zonder kennis van buitenaf en
  zonder verzinsels?
  1 = bevat verzonnen beweringen, of beweringen die het artikel tegenspreken;
  3 = grotendeels gesteund, maar met een of meer details die nergens in het
  artikel staan (ook als ze in werkelijkheid kloppen);
  5 = elke bewering is expliciet door het artikel gesteund.
- **fluency**: hoe vloeiend, natuurlijk en grammaticaal correct is het
  Nederlands van vraag en antwoord samen (spelling, zinsbouw,
  werkwoordsvormen, woordkeuze)?
  1 = houterig of duidelijk machinaal, met veel fouten;
  3 = begrijpelijk, maar met stroeve formuleringen of een paar fouten;
  5 = foutloos en leest als door een moedertaalspreker geschreven.
- **coherence**: sluit het antwoord logisch en inhoudelijk aan bij de vraag?
  1 = het antwoord gaat niet over de vraag;
  3 = beantwoordt de vraag maar gedeeltelijk, of dwaalt af;
  5 = beantwoordt precies en volledig wat er gevraagd wordt.

Beoordeel wat er staat, niet wat er bedoeld lijkt te zijn. Wees streng: dit
oordeel bepaalt of het model waarvan dit paar afkomstig is, betrouwbaar
genoeg is om op grote schaal trainingsdata mee te genereren.

Titel: {title}

Artikel:
{text}

Vraag:
{question}

Antwoord:
{answer}
