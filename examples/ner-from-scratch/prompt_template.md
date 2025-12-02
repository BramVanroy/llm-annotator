**Task:**
Generate **20 sentences** annotated for Named Entity Recognition (NER) in IOB2 format.

**Entity types allowed:**

* PER: person names (real or fictional person names)
* ORG: organizations (any named collection of people, such as firms, institutions, organizations, artists, sports teams, political parties etc.)
* LOC: locations (airports, restaurants, hotels, tourist attractions, shops, street addresses, oceans, fjords, planets, parks and fictional locations)

**Important annotation rules:**

* Tag only **named entities** that meet the UNER criteria: they are proper nouns or include proper nouns; they refer to a unique entity with a constant reference.
* Use **B-TYPE** to mark the **first token** of an entity span, and **I-TYPE** for subsequent tokens of the same span; all other tokens get **O**.
* Strictly follow IOB2 format: I-TYPE tags should always be preceded by a B-TYPE or I-TYPE of the same type to constitute a single entity span.
* Entities must be **flat spans**, never nested. E.g., annotate [University of Washington St. Louis](ORG), not [University of Washington](LOC) [St. Louis](LOC).
* In case of ambiguity (e.g., same string could be ORG or LOC), choose the **literal meaning** or the **most common usage** given the context.
* Do *not* annotate:
  * Time expressions
  * Nationalities or languages or adjectives derived from them
* For LOC: include geographical places, buildings, facilities, street addresses, etc. E.g., [The Gulf of Mexico](LOC)
* For ORG: include named collections of people, institutions, sports teams, etc. Corporate designators (Co., Ltd.) are part of the ORG name.
* For PER: include individual people (real or fictional), including names with initials, nicknames, etc. E.g., [Mr. Grinch](PER), [Charlie Chaplin](PER).

**Output format:**
For each sentence:

1. Start with a line beginning with `# ` followed by the sentence.
2. On the following lines, list **each token** of the sentence followed by a tab character `\t` then the IOB2 tag. One token per line.

**Additional guidance:**

* Include **variation** in topics (blogs, newsgroups, emails, reviews, Q&A), sentence length, syntax, and word order.
* Use both **single-token entities** (just B-TYPE) and **multi-token entities** (B-TYPE + I-TYPE).
* Do *not* include any extra explanation, only the annotated examples.

**Examples:**

# Crude-oil prices rose Wednesday as strengthening Hurricane Rita, now a Category 5 storm, threatened to disrupt oil production in the Gulf of Mexico.
Crude	O
-	O
oil	O
prices	O
rose	O
Wednesday	O
as	O
strengthening	O
Hurricane	O
Rita	O
,	O
now	O
a	O
Category	O
5	O
storm	O
,	O
threatened	O
to	O
disrupt	O
oil	O
production	O
in	O
the	O
Gulf	B-LOC
of	I-LOC
Mexico	I-LOC
.	O

# There 's also a Miramar in California, the site of a rather large Air Force Base...
There	O
's	O
also	O
a	O
Miramar	B-LOC
in	O
California	B-LOC
,	O
the	O
site	O
of	O
a	O
rather	O
large	O
Air	B-LOC
Force	I-LOC
Base	I-LOC
...	O

# Marlene Hilliard
Marlene	B-PER
Hilliard	I-PER

# i 'm doing a report on how afghanistan and Vietam are different and alike.
i	O
'm	O
doing	O
a	O
report	O
on	O
how	O
afghanistan	B-LOC
and	O
Vietam	B-LOC
are	O
different	O
and	O
alike	O
.	O

# Once upon a time (in 2001, to be specific), the Coca-Cola corporation built a bottling plant in a small and remote Indian village in the state of Kerala.
Once	O
upon	O
a	O
time	O
(	O
in	O
2001	O
,	O
to	O
be	O
specific	O
)	O
,	O
the	O
Coca	B-ORG
-	I-ORG
Cola	I-ORG
corporation	O
built	O
a	O
bottling	O
plant	O
in	O
a	O
small	O
and	O
remote	O
Indian	O
village	O
in	O
the	O
state	O
of	O
Kerala	B-LOC
.	O

# As such, it is essential that HANO comply with 2003 enforcement agreement," said James Perry, GNOFHAC Executive Director
As	O
such	O
,	O
it	O
is	O
essential	O
that	O
HANO	B-ORG
comply	O
with	O
2003	O
enforcement	O
agreement	O
,	O
"	O
said	O
James	B-PER
Perry	I-PER
,	O
GNOFHAC	B-ORG
Executive	O
Director	O

