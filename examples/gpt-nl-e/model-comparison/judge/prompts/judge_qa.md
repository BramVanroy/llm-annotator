Beoordeel dit Nederlandstalige vraag-antwoordpaar, dat automatisch is
gegenereerd op basis van het Wikipedia-artikel hieronder. Dit paar is bedoeld
als trainingsdata voor een taalmodel; jouw beoordeling bepaalt of het
bruikbaar is.

- **question_relevant** — gaat de vraag daadwerkelijk over de inhoud van het
  artikel, en peilt ze naar een concreet feit, gebeurtenis, definitie of
  verband (dus niet naar iets triviaals of iets ernaast)?
- **question_self_contained** — staat de vraag op zichzelf? False zodra ze
  verwijst naar "dit artikel", "de tekst hierboven", "bovenstaande" of iets
  van gelijke strekking. Iemand die het artikel nooit gezien heeft, moet de
  vraag kunnen lezen en begrijpen waar ze over gaat.
- **answer_correct** — is het antwoord feitelijk juist volgens het artikel?
- **answer_grounded** — steunt het antwoord uitsluitend op wat er letterlijk
  of logisch afleidbaar in het artikel staat, zonder kennis van buitenaf?
- **hallucinated** — bevat het antwoord een bewering die niet in het artikel
  staat, of die het artikel tegenspreekt? True zodra er ook maar één detail
  is verzonnen of fout overgenomen, ook als de rest van het antwoord klopt.
- **fluency** — hoe vloeiend en natuurlijk is het Nederlands van vraag en
  antwoord samen, op een schaal van 1 (houterig, duidelijk machinaal) tot 5
  (leest als door een moedertaalspreker geschreven)?
- **coherence** — sluit het antwoord logisch en inhoudelijk aan bij de vraag,
  op een schaal van 1 (antwoord gaat niet over de vraag) tot 5 (perfect
  aansluitend)?
- **grammar** — hoe correct is de grammatica (spelling, zinsbouw,
  werkwoordsvormen), op een schaal van 1 (veel fouten) tot 5 (foutloos)?
- **overall_quality** — algemene bruikbaarheid van dit paar als trainingsdata,
  op een schaal van 1 (onbruikbaar) tot 5 (uitstekend). Dit cijfer weegt alle
  bovenstaande criteria samen, met een zwaarder gewicht voor
  `answer_correct`, `answer_grounded` en `hallucinated`: een fout of verzonnen
  antwoord kan nooit hoger dan 2 scoren, ongeacht hoe vloeiend het geschreven
  is.

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
