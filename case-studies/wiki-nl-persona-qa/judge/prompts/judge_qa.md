Beoordeel een Nederlandstalig vraag-antwoordpaar dat automatisch is gegenereerd
op basis van het Wikipedia-artikel hieronder. De vraag hoort te klinken als een
vraag die iemand uit zichzelf aan een AI-assistent stelt; het antwoord hoort
volledig op dat artikel te steunen. Dit paar is bedoeld als trainingsdata voor
een taalmodel; jouw beoordeling bepaalt of het bruikbaar is.

Geef elk criterium een geheel getal van 1 tot en met 5. Gebruik de volledige
schaal: 3 is het normale oordeel voor "grotendeels in orde, maar met een
duidelijk mankement", en 5 is voorbehouden aan werk waar niets op aan te merken
valt. Beoordeel elk criterium apart; laat een zwakke score op het ene criterium
niet doorwegen op het andere.

- **question_answerable**: bevat het artikel alles wat nodig is om de vraag te
  beantwoorden?
  1 = het artikel zegt hier niets over, de vraag gaat over iets anders;
  3 = het artikel beantwoordt de vraag maar half, of alleen indirect;
  5 = het artikel bevat alles wat voor een volledig antwoord nodig is.
- **question_self_contained**: staat de vraag op zichzelf? Iemand die het
  artikel nooit gezien heeft, moet ze kunnen lezen en meteen weten waarover ze
  gaat.
  1 = verwijst naar "dit artikel", "de tekst hierboven", "bovenstaande" of iets
  van gelijke strekking, en is zonder het artikel betekenisloos;
  3 = geen expliciete verwijzing, maar mist context (bv. een "hij" of "het
  bedrijf" zonder dat duidelijk is wie of wat bedoeld wordt);
  5 = volledig zelfstandig leesbaar, met het onderwerp voluit benoemd.
- **question_natural**: klinkt dit als een vraag die iemand uit interesse aan
  een assistent stelt?
  1 = een examenvraag over een tekst ("Welke drie oorzaken worden genoemd?"),
  of een geforceerde formulering;
  3 = een redelijke vraag, maar stijf of schools geformuleerd;
  5 = klinkt als een echte, spontaan gestelde vraag.
- **answer_correct**: is het antwoord feitelijk juist volgens het artikel?
  1 = spreekt het artikel tegen, of is gewoon fout;
  3 = de kern klopt, maar een detail (getal, datum, naam) is fout of
  onnauwkeurig;
  5 = volledig juist volgens het artikel.
- **answer_grounded**: steunt het antwoord uitsluitend op wat er letterlijk of
  logisch afleidbaar in het artikel staat, zonder kennis van buitenaf en zonder
  verzinsels?
  1 = bevat verzonnen beweringen, of beweringen die het artikel tegenspreken;
  3 = grotendeels gesteund, maar met een of meer details die nergens in het
  artikel staan (ook als ze in werkelijkheid kloppen);
  5 = elke bewering is expliciet door het artikel gesteund.
- **answer_complete**: beantwoordt het antwoord precies en volledig wat er
  gevraagd wordt?
  1 = het antwoord gaat niet over de vraag;
  3 = beantwoordt de vraag maar gedeeltelijk, dwaalt af, of verdrinkt de kern
  in overbodige context;
  5 = beantwoordt de vraag volledig, zonder opvulling.
- **fluency**: hoe vloeiend, natuurlijk en grammaticaal correct is het
  Nederlands van vraag en antwoord samen (spelling, zinsbouw, werkwoordsvormen,
  woordkeuze)?
  1 = houterig of duidelijk machinaal, met veel fouten;
  3 = begrijpelijk, maar met stroeve formuleringen of een paar fouten;
  5 = foutloos en leest als door een moedertaalspreker geschreven.

Beoordeel wat er staat, niet wat er bedoeld lijkt te zijn. Wees streng: dit
oordeel bepaalt of dit paar in de trainingsset terechtkomt.

Titel: {title}

Artikel:
{text}

Vraag:
{question}

Antwoord:
{answer}
