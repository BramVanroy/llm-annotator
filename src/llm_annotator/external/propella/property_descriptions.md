## Detailed Property Descriptions & Annotation Guidelines

### Core Content Properties

#### 1. Content Integrity

**What we're measuring**: Completeness and technical quality of the content itself, regardless of navigation ratio.

##### Values & Criteria:

**`complete`** - Full, intact content as intended
- Content appears complete with proper beginning, middle, and end
- All essential elements present (introduction, body, conclusion where appropriate)
- No obvious truncation or missing sections
- Example: Complete articles, full tutorials, intact documents

**`mostly_complete`** - Minor elements missing but core content intact
- Core content is complete but some secondary elements may be missing
- Minor truncation that doesn't affect main message
- Example: Article with truncated comments, missing sidebar content, partial author bio

**`fragment`** - Incomplete content, missing significant portions
- Missing introduction, conclusion, or substantial middle sections
- Truncated mid-sentence or mid-paragraph
- Content feels incomplete or cut off
- Example: Search result snippets, article excerpts, broken crawls, partial downloads

**`severely_degraded`** - Broken, unreadable, or corrupted content
- Encoding errors, scrambled text, missing characters
- Severely malformed HTML rendering as gibberish
- Technical corruption making content unreadable
- Example: �&$^%*@# characters, completely broken formatting, corrupted files

##### Key Decision Points:
- **Content completeness**: Does the content feel like a complete unit of information?
- **Technical integrity**: Is the content technically readable and properly formatted?
- **Fragment vs. complete**: Independent of navigation - is the actual content complete?
- **Degraded vs. fragment**: Degraded has technical issues; fragment is just incomplete

**Note**: Documents may end with the special tag `<content_truncated>`, indicating upstream length-based truncation due to processing constraints. Do not penalize Content Integrity due to this truncation signal; assess integrity based on the visible content's coherence and technical readability, ignoring the artificial cutoff.

---

#### 2. Content Ratio

**What we're measuring**: How much of the document is actual content vs. navigation, UI elements, and structural markup.

##### Values & Criteria:

**`complete_content`** - 90-100% meaningful content
- Full articles, papers, tutorials with minimal navigation
- Clean text with proper paragraphs and structure
- Example: A Wikipedia article, academic paper, complete blog post

**`mostly_content`** - 70-89% meaningful content  
- Complete documents with some navigation elements (header, footer, sidebar)
- Minor UI elements that don't disrupt reading
- Example: News articles with standard website navigation

**`mixed_content`** - 40-69% meaningful content
- Significant navigation mixed throughout content
- Multiple sidebars, ads, or UI elements interrupting text
- Example: E-commerce product pages with reviews mixed with purchase options

**`mostly_navigation`** - 10-39% meaningful content
- Predominantly menus, links, headers, footers
- Content overwhelmed by structural elements
- Example: Site maps, navigation pages, heavily UI-focused pages

**`minimal_content`** - 0-9% meaningful content
- Almost entirely navigation, UI elements, or structural markup
- Very little readable content present
- Example: Empty pages, pure navigation menus, error pages with minimal text

##### Key Decision Points:
- Focus on the **ratio of readable text to navigation/UI elements**
- **Count only substantive content**, ignore boilerplate and structural elements
- **Mixed vs. mostly_navigation**: Can you read it as coherent content despite distractions?

---

#### 3. Content Length

**What we're measuring**: Amount of substantive content, ignoring navigation and boilerplate.

##### Values & Criteria:

**`substantial`** - 2,000+ words of meaningful content
- Long-form, comprehensive content that provides in-depth coverage of a topic
- Typically includes detailed analysis, multiple sections or chapters, extensive research, or thorough exploration of complex subjects
- Examples: White papers, research reports, e-books, long-form journalism

**`moderate`** - 5002,000 words of meaningful content
- Standard-length content that offers meaningful coverage while remaining focused and digestible
- Balances depth with accessibility; provides enough detail to be informative without overwhelming readers
- Examples: Typical blog posts, news articles, product reviews, how-to guides

**`brief`** - 100500 words of meaningful content
- Short, focused content that delivers key information quickly and efficiently
- Gets straight to the point while still providing value and context
- Examples: News briefs, product descriptions, FAQs, short blog posts

**`minimal`** - Under 100 words of meaningful content
- Very short content that provides only essential information or serves as a quick reference
- Designed for rapid consumption or specific micro-purposes
- Examples: Social media posts, announcements, abstracts, snippets, navigation pages

##### Measurement Tips:
- **Count only readable content of value**: include article body and substantive headings/captions; exclude headers/footers, menus/sidebars, related links, share/consent UI, pagination, ads, and boilerplate.
- **Focus on substantive information**, not filler words
- **Complete thoughts matter more than exact word counts**
- **Contextual adjustment**: Thresholds are guidelines and can be adjusted based on specific use cases and typical content. Academic contexts may shift ranges upward, while social media contexts may shift them downward.

---

### Content Classification

#### 4. One-Sentence Description

**What we're looking for**: A very short, neutral description of what the document contains.

##### Field:

**`one_sentence_description`**
- Ultra-short neutral description of the document
- Exactly one sentence
- Target length: <100 characters
- Focus on the main topic and, if useful, the document’s function
- Examples of functions: tutorial, policy, news report, product page, navigation page
- Neutral, descriptive tone (no hype or marketing language)

##### To Avoid:
- Boilerplate intros: "This document...", "This article...", "In this guide..."
- Calls to action: "Learn how to...", "Discover...", "Find out..."
- User-facing phrasing: "You will learn...", "How do I..."
- Non-essential details (dates, numbers) unless central to the topic

##### Examples:
- "Beginner tutorial on React hooks and basic state management."
- "News report on European Central Bank interest rate decisions."
- "Internal policy for customer data retention and deletion."
- "API reference for payment processing endpoints and error codes."
- "Research paper analyzing housing price trends in major US cities."
- "FAQ answering common questions about employee parental leave."
- "Opinion essay arguing for stricter international climate change legislation."

##### Examples for low-quality or problematic documents (still annotate):
- "Fragment of article discussing proposed changes to European data privacy laws."
- "Keyword-stuffed promotional page about cheap car insurance quotes."
- "Website navigation page listing links to product categories and help pages."
- "Error page explaining that the requested resource could not be found."
- "Affiliate landing page promoting multiple online casino bonus offers."
- "Corrupted text with no identifiable topic or meaningful content."


#### 5. Content Type

**What we're measuring**: The functional structure and purpose of content.

**Multi-type content**: Content can be assigned multiple type labels if it genuinely serves multiple purposes. Choose ALL applicable types rather than forcing a single primary choice. Always output an array for this property, even if only one type applies.

##### Values & Criteria:

**`analytical`** - In-depth analysis, research, and critical examination
- Provides detailed analysis or research on a topic
- Develops arguments, evaluates evidence, or presents findings
- Example: Research analysis, investigative reports, academic articles, expert commentary

**`instructional`** - Teaching and how-to content
- Explicitly teaches skills, concepts, or procedures
- Step-by-step guidance or educational explanations
- Example: Tutorials, how-to guides, educational content, training materials

**`reference`** - Lookup materials, definitions, specifications
- Designed for looking up specific information rather than reading through
- Often organized alphabetically, categorically, or as lists
- Example: Dictionaries, encyclopedias, API references, product catalogs

**`procedural`** - Step-by-step processes and procedures
- Sequential instructions or workflows
- Process documentation with clear steps
- Example: Recipes, installation guides, standard operating procedures, workflows

**`qa_structured`** - Structured question-answer content
- Formal Q&A format with clear questions and answers
- Often expert responses to specific questions
- Example: Stack Overflow, FAQ sections, structured Q&A sites

**`conversational`** - Multi-party or turn-based dialogues (humans, bots, or both)
- Casual or structured conversations between two or more participants
- May include humanAI chats, forum threads, or comment chains
- Example: Reddit threads, forum discussions, support chats, assistant chat logs

**`creative`** - Entertainment, artistic, fictional content
- Primary purpose is entertainment or artistic expression
- Not primarily informational or instructional
- Example: Short stories, poems, movie reviews, game content, fiction

**`transactional`** - Commercial, shopping, service-oriented
- Primary purpose is to facilitate a transaction or service
- Focuses on products, services, or business processes
- Example: Product listings, service descriptions, checkout pages

**`boilerplate`** - Legal, policy, standard template text
- Standard legal or policy language
- Often repeated across multiple sites with minimal variation
- Example: Terms of service, privacy policies, disclaimers, cookie banners, standard notices

**`news_report`** - Straight reporting of events with minimal analysis
- Describes events or facts in a neutral, descriptive tone
- Time-bound news, updates, or reports
- Example: Wire-service news articles, breaking-news updates

**`opinion_editorial`** - Persuasive/opinionated commentary or editorials
- Expresses a stance or argument; aims to persuade
- May cite evidence but prioritizes viewpoint
- Example: Op-eds, opinion columns, personal essays with clear stance

**`review_critique`** - Evaluative reviews of products, media, or services
- Provides judgments, ratings, or critiques
- May include pros/cons, scoring systems
- Example: Product reviews, film/book critiques, app store reviews (long-form)

**`technical_documentation`** - Manuals, API docs, developer guides, READMEs
- Primary goal is to instruct usage of software/hardware/APIs
- Includes reference sections, examples, parameters, version notes
- Example: API reference, library README, user manual

**`specification_standard`** - Normative standards and formal specifications
- Defines requirements, must/shall language, compliance criteria
- Maintained by standards bodies or authoritative groups
- Example: RFCs, ISO standards, formal protocol specs

**`legal_document`** - Statutes, case law, contracts, regulatory texts
- Binding or authoritative legal content
- Formal legal language and structure
- Example: Court opinions, legislation, contracts, regulatory rules

**`press_release`** - Organization-issued announcements and PR materials
- Promotional announcements framed as information
- Quotes from executives, product/service announcements
- Example: Company press releases, launch announcements


**`structured_data`** - Tables, datasets, indices, catalogs with minimal prose
- Predominantly tabular/listed data meant for lookup
- Minimal narrative or explanatory text
- Example: Product catalogs, schedules, statistical tables

**`source_code`** - Code listings as primary content
- Dominant content is program source code or scripts
- May include lightweight comments or snippets without narrative
- Example: Code files, gist-like pages, competitive programming solutions


##### Multi-Type Examples:
- **Tutorial that analyzes different approaches** → `["instructional", "analytical"]`
- **Educational reference manual** → `["instructional", "reference"]`
- **Research paper with step-by-step methodology** → `["analytical", "procedural"]`
- **Q&A site with analytical responses** → `["qa_structured", "analytical"]`
- **API guide with examples** → `["technical_documentation", "reference", "instructional"]`
- **RFC with rationale** → `["specification_standard", "analytical"]`
- **Film review with interview snippets** → `["review_critique", "conversational"]`
- **Helpdesk chat with an AI** → `["conversational", "transactional"]`
- **Breaking news explainer** → `["news_report", "explanatory"]`

---

#### 6. Business Sector

**What we're measuring**: Business sector(s) or industry domain(s) for training sector-specific LLMs.

**Multi-sector content**: Content can be assigned multiple sector labels if it genuinely spans multiple industries. Choose ALL applicable sectors rather than forcing a single primary choice or using "other". Always output an array for this property, even if only one sector applies.

#### Values & Criteria:

**`academic_research`** - Scholarly and research content
- Peer-reviewed publications, academic papers
- University-affiliated research and scholarship
- Formal academic discourse and methodology
- Example: Journal articles, conference papers, academic books, dissertations

**`education_sector`** - Educational institutions and pedagogy
- K-12 education, higher education administration
- Educational technology, curriculum development
- Teaching methodologies and educational policy
- Example: School curricula, educational policy papers, teaching resources, edtech content

**`technology_software`** - Software and information technology
- Software development, programming, IT services
- Digital products, platforms, and technology companies
- Computer science and software engineering
- Example: Software documentation, tech company content, programming guides, IT industry analysis

**`hardware_electronics`** - Hardware devices and electronics industry
- Semiconductors, consumer electronics, embedded systems, hardware design
- Electronics manufacturing and supply chains
- Example: Chip design docs, hardware datasheets, device manuals

**`healthcare_medical`** - Healthcare and medical sector
- Medical research, clinical practice, healthcare delivery
- Hospitals, medical devices, healthcare policy
- Public health and wellness
- Example: Medical journals, clinical guidelines, healthcare administration, wellness content

**`pharmaceutical_biotech`** - Pharmaceutical and biotechnology
- Drug development, clinical trials, biotech research
- Pharmaceutical industry, biotechnology companies
- Life sciences and molecular biology applications
- Example: Drug research papers, clinical trial reports, biotech industry analysis

**`financial_services`** - Banking and financial services
- Banking, investment, insurance, financial planning
- Financial markets, fintech, payment systems
- Asset management and financial advisory
- Example: Financial analysis, banking documentation, investment guides

**`legal_services`** - Legal sector and jurisprudence
- Law firms, legal practice, court systems
- Legal education, regulatory compliance
- Litigation, contracts, legal advisory
- Example: Legal briefs, court opinions, legal analysis, compliance guides

**`government_public`** - Government and public administration
- Government agencies, public policy, civic services
- Regulatory bodies, public administration
- Political institutions and governance
- Example: Government reports, policy documents, regulatory filings, civic information

**`manufacturing_industrial`** - Manufacturing and heavy industry
- Industrial production, manufacturing processes
- Supply chain, logistics, industrial equipment
- Factory operations and industrial engineering
- Example: Manufacturing specs, industrial reports, supply chain analysis, production guides

**`mining_resources`** - Mining and natural resources
- Exploration, extraction, and processing of minerals and resources
- Resource markets and operations (metals, rare earths)
- Example: Mining reports, resource exploration docs, commodity operations

**`chemicals_materials`** - Chemicals and advanced materials
- Petrochemicals, specialty chemicals, polymers, composites, advanced materials
- Safety data sheets (SDS), process chemistry, materials science
- Example: Material datasheets, REACH documentation, chemical process guides

**`energy_utilities`** - Energy and utilities sector
- Power generation, renewable energy, oil and gas
- Electric utilities, water services, waste management
- Energy infrastructure and grid management
- Example: Energy industry reports, utility regulations, renewable energy research

**`retail_commerce`** - Retail and e-commerce
- Retail operations, e-commerce platforms
- Consumer goods distribution, merchandising
- Retail technology and customer experience
- Example: Retail industry analysis, e-commerce guides, merchandising strategies

**`wholesale_distribution`** - Wholesale trade and distribution
- B2B wholesale, distributors, procurement, inventory and fulfillment
- Supply relationships between manufacturers and retailers
- Example: Distributor catalogs, wholesale operations, procurement guides

**`real_estate_construction`** - Real estate and construction
- Property development, construction industry
- Real estate markets, property management
- Architecture and building services
- Example: Real estate analysis, construction specifications, property guides

**`transportation_logistics`** - Transportation and logistics
- Airlines, shipping, freight, public transit
- Logistics operations, supply chain transportation
- Vehicle fleet management, transportation infrastructure
- Example: Logistics guides, transportation planning, shipping documentation

**`travel_aviation`** - Travel industry and commercial aviation
- Airlines, airports, OTA platforms, hospitality travel operations
- Route planning, airline commercial, loyalty, IATA regulations
- Example: Airline scheduling, fare rules, OTA partner docs

**`automotive_industry`** - Automotive manufacturing and services
- Vehicle manufacturers, automotive suppliers
- Automotive technology, electric vehicles
- Dealerships and automotive services
- Example: Automotive engineering docs, vehicle technology papers, industry analysis

**`telecommunications`** - Telecommunications industry
- Telecom operators, network infrastructure
- Mobile services, broadband, satellite communications
- Telecommunications equipment and technology
- Example: Telecom industry reports, network specifications, 5G technology papers

**`media_entertainment`** - Media and entertainment industry
- Film, television, music, gaming industries
- Publishing, news media, content creation
- Streaming services and digital media
- Example: Entertainment industry analysis, media studies, content strategy

**`gaming_industry`** - Video games and interactive entertainment
- Game development, studios, engines, esports, live ops
- Monetization models, community management, platform ecosystems
- Example: Patch notes, game design docs, esports operations

**`gambling_betting`** - Gambling, betting, and online casinos
- Sportsbooks, casino games, lotteries, poker rooms
- Affiliate landing pages, bonus/promotions, tipster content
- Often high commercial bias and promotional framing

**`advertising_marketing`** - Advertising, marketing, and PR
- Brand strategy, campaign planning, performance marketing, martech
- Agencies, in-house marketing, PR communications
- Example: Campaign briefs, media plans, PR strategies

**`hospitality_tourism`** - Hospitality and tourism sector
- Hotels, restaurants, travel services
- Tourism industry, destination management
- Event planning and hospitality services
- Example: Tourism studies, hospitality management, travel industry reports

**`food_beverage_hospitality`** - Food & beverage and restaurant operations
- Restaurant ops, menu engineering, supply chain, QSR/fast casual
- Food safety, compliance, procurement for F&B
- Example: Restaurant training manuals, HACCP docs, vendor specs

**`agriculture_food`** - Agriculture and food production
- Farming, agricultural technology, food processing
- Agricultural supply chain, food safety
- Agribusiness and agricultural policy
- Example: Agricultural research, food industry reports, farming guides

**`environmental_services`** - Environmental and sustainability services
- Environmental consulting, ESG reporting, sustainability programs
- Waste management services, remediation, impact assessments
- Example: ESG reports, environmental impact assessments, sustainability frameworks

**`aerospace_defense`** - Aerospace and defense industry
- Aircraft manufacturing, space technology
- Defense contractors, military systems
- Aviation and space exploration
- Example: Aerospace engineering papers, defense industry analysis, aviation guides

**`insurance_industry`** - Insurance sector
- Life, health, property, and casualty insurance
- Reinsurance, actuarial science, risk assessment
- Insurance technology and underwriting
- Example: Actuarial studies, insurance policy analysis, risk management guides

**`nonprofit_ngo`** - Nonprofit and NGO sector
- Charitable organizations, international development
- Social services, humanitarian organizations
- Foundations and philanthropic institutions
- Example: NGO reports, nonprofit management, development studies

**`consulting_professional`** - Professional services and consulting
- Management consulting, accounting firms
- Business advisory, professional services firms
- Corporate strategy and business transformation
- Example: Consulting reports, professional services guides, business strategy papers

**`human_resources`** - HR and people operations
- Talent acquisition, compensation & benefits, performance management, L&D
- HR tech, workforce planning, organizational development
- Example: HR policy docs, job frameworks, talent strategy

**`security_cyber`** - Security and cybersecurity
- Information security, threat intelligence, risk management, compliance (e.g., SOC2)
- Physical security operations and incident response
- Example: Security guidelines, incident playbooks, vulnerability reports

**`consumer_goods`** - Consumer products and CPG
- Fast-moving consumer goods, household products
- Personal care, food and beverage brands
- Consumer product development and marketing
- Example: CPG industry analysis, product development docs, consumer research

**`general_interest`** - General audience content
- Content for broad audiences without sector focus
- General knowledge and miscellaneous topics
- Cross-sector or sector-agnostic content
- Example: General magazines, broad interest content, lifestyle articles

**`other`** - Highly specialized or unclassifiable
- Highly specialized niches not covered by existing sectors
- Content with genuinely unclear sector classification
- Unique content types that don't map to any defined sector
- Example: Highly specialized technical niches, unique content formats

##### Multi-Sector Examples:
- **Medical device regulations** → `healthcare_medical` + `pharmaceutical_biotech` + `government_public`
- **Fintech software documentation** → `financial_services` + `technology_software`
- **Agricultural biotechnology research** → `agriculture_food` + `pharmaceutical_biotech`

---

#### 7. Technical Content

**What we're measuring**: Type and intensity of specialized technical knowledge.

**Multi-technical content**: Content can be assigned multiple technical content labels if it genuinely combines multiple technical domains. Choose ALL applicable technical types rather than forcing a single primary choice. Always output an array for this property, even if only one technical type applies.

##### Values & Criteria:

**`code_heavy`** - Significant programming content
- Multiple code examples, algorithms, or implementations
- Technical programming concepts and methodologies
- Software development focus
- Example: Programming tutorials, API documentation, software guides

**`math_heavy`** - Substantial mathematical content
- Mathematical equations, proofs, or statistical analysis
- Quantitative analysis and mathematical reasoning
- Mathematical concepts and methodologies  
- Example: Mathematical papers, statistical analysis, quantitative research

**`scientific`** - Research and scientific methodology content
- Scientific research findings, experimental data
- Scientific methodology and analysis
- Peer-reviewed research content
- Example: Research papers, scientific studies, experimental reports

**`data_heavy`** - Substantial datasets, tables, and data analysis
- Contains significant data tables, charts, or datasets
- Focus on data interpretation and analysis
- Statistical content with data presentations
- Example: Research data, statistical reports, data analysis, survey results

**`engineering`** - Engineering and applied technical content
- Engineering design, systems, and applied technical solutions
- Technical specifications for physical systems
- Non-software engineering disciplines
- Example: Mechanical engineering, civil engineering, technical specifications, design documents

**`basic_technical`** - Some technical elements but not dominant
- Light technical content mixed with general explanations
- Technical concepts explained for general audience
- Example: Technology articles for general audience, basic technical explanations

**`non_technical`** - No significant technical content
- General audience content without specialized technical knowledge
- No programming, mathematical, engineering, or scientific focus
- Example: General articles, humanities content, basic informational content

##### Multi-Technical Examples:
- **Data science tutorial with code examples** → `["code_heavy", "math_heavy", "data_heavy"]`
- **Engineering research with statistical analysis** → `["engineering", "scientific", "data_heavy"]`
- **Computational biology paper** → `["code_heavy", "scientific"]`

---

### Quality and Value Assessment

#### 8. Content Quality

**What we're measuring**: Overall quality of content considering writing excellence, substantive value, and presentation quality regardless of authorship origin.

#### Values & Criteria:

**`excellent`** - Outstanding quality across all dimensions
- Sophisticated writing with varied sentence structures and engaging style
- Rich, appropriate vocabulary with error-free grammar and punctuation
- High substantive value with clear insights or information
- Professional presentation and formatting
- Natural flow and logical organization
- Example: High-quality publications, expert analyses, polished educational content, well-crafted professional documents

**`good`** - High quality with minor imperfections
- Grammatically correct with proper sentence structure
- Appropriate vocabulary and tone for content type
- Solid substantive value and clear information
- Good organization and readable flow
- Only occasional minor issues (1-2 typos per section)
- Example: Quality journalism, professional websites, well-written blog posts, solid educational materials

**`adequate`** - Acceptable quality for most purposes
- Generally clear and understandable writing
- Some grammatical errors but meaning remains clear
- Reasonable substantive value though may lack depth
- Basic organization and structure present
- Minor formatting or presentation issues
- Example: Casual blogs, user reviews, basic informational content, simple guides

**`poor`** - Significant quality issues impacting utility
- Multiple errors affecting comprehension or credibility
- Unclear expression, confusing organization, or awkward phrasing
- Limited substantive value or questionable information
- Major formatting problems or unprofessional presentation
- Difficult to extract reliable information
- Example: Low-quality web content, poorly edited materials, confusing instructions

**`unacceptable`** - Quality too low for productive use
- Severely impaired communication with major errors
- Incoherent, nonsensical, or corrupted content
- No reliable substantive value
- Broken formatting or technical corruption
- Cannot determine intended meaning or extract useful information
- Example: Corrupted text, severe translation errors, spam content, SEO content, completely broken formatting

##### Quality Assessment Guidelines:
- **Comprehension**: Can the intended message be clearly understood?
- **Substantive value**: Does the content provide useful information or insights?
- **Technical presentation**: Is the content properly formatted and readable?
- **Error impact**: Do errors significantly impede understanding or credibility?
- **Professional standards**: Does the content meet basic standards for its intended purpose?

**Language-Specific Quality Indicators:**
- For non-Latin scripts (Arabic, Chinese, Japanese): Check for proper character encoding
- For agglutinative languages (Turkish, Finnish): Adjust expectations for word count/density
- For languages with different formality levels (Japanese, Korean): Assess appropriate register
- Mixed-language documents: Evaluate code-switching quality and appropriateness
---

#### 9. Information Density

**What we're measuring**: Ratio of valuable information to redundancy, padding, and repetition.

##### Values & Criteria:

**`dense`** - Efficient, information-packed content
- Every sentence adds new information or insight
- Minimal redundancy or unnecessary elaboration
- Little to no repetition of the same concepts
- Example: Technical specifications, concise academic writing, quality reference material

**`adequate`** - Good information content with reasonable elaboration
- Most content adds value with some acceptable elaboration
- Minimal repetition within the document
- Good balance of information and explanation
- Example: Well-written articles, good tutorials with examples

**`moderate`** - Mixed substantive content with noticeable padding
- Some valuable information mixed with unnecessary elaboration
- Noticeable repetition of key points for emphasis
- Some sections feel padded or verbose
- Example: Blog posts with some fluff, articles with repetitive conclusions

**`thin`** - Low information content with significant problems
- Much content doesn't add new information
- High internal repetition and excessive redundancy  
- Significant padding to reach desired length
- Example: SEO-optimized content, poorly edited writing

**`empty`** - Dominated by repetition and meaningless content
- Minimal actual information value
- Dominated by repetition and copy-paste artifacts
- Same ideas repeated multiple times without development
- Example: Spam content, template-filled pages, keyword-stuffed articles

##### Common Repetition Patterns to Watch For:
- **Same phrases repeated throughout** (especially in SEO content)
- **Identical paragraphs** or sections (copy-paste errors)
- **Circular reasoning** (saying the same thing in different ways)
- **Template artifacts** (repeated boilerplate mixed with content)

---

#### 10. Educational Value

**What we're measuring**: Potential for teaching, learning, and knowledge transfer.

##### Values & Criteria:

**`high`** - Clear instructional design and learning objectives
- Explicitly teaches concepts or skills
- Progressive skill building from basic to advanced
- Clear learning objectives and outcomes
- Comprehensive explanations with examples
- Example: Quality tutorials, textbooks, structured courses, educational guides

**`moderate`** - Good instructional value with some learning potential
- Some instructional elements present
- Explanations help build understanding
- Transferable knowledge to other contexts
- Good examples or illustrations
- Example: How-to articles, explanatory content, informative guides

**`basic`** - Limited educational content
- Some explanations but not systematically instructional
- Basic explanations of concepts
- Limited learning potential or skill building
- Example: Basic explanations, simple informational content

**`minimal`** - Little educational value
- Primarily informational rather than instructional
- No clear learning objectives or skill building
- Entertainment or commercial focus
- Example: Entertainment content, basic news, commercial content

**`none`** - No educational content
- No instructional value or learning potential
- Purely transactional, entertainment, or administrative
- No knowledge transfer potential
- Example: Pure entertainment, transactions, legal boilerplate

##### Disambiguation tips
- Explanatory vs Educational: explanations alone ≠ educational design; require intent to teach plus scaffolding for Basic+
- Reference docs: typically Minimal; promote to Basic/Moderate when guided “how-to” segments or curated examples exist
- Reviews/op-eds: None/Minimal unless they include actionable how-to guidance designed for learning

##### Automation heuristics
- Keywords: Objectives/Outcomes, Lesson, Exercise/Quiz, Homework, Assessment, Syllabus, Module, Unit, Learning Goals
- Structure: numbered steps + prerequisites/requirements → Basic; add practice tasks/solutions → Moderate; syllabus/modules/assessments → High
- Signals of non-edu mix: heavy CTAs/ads or product pitches → cap at Minimal unless clear instructional scaffolding

##### Quick decision tree
- Are there explicit learning goals or a syllabus? → High
- Else, are there step-by-step instructions with examples/exercises? → Moderate
- Else, are there explanatory sections intended to teach basics? → Basic
- Else, is there any minor instructional element? → Minimal
- Otherwise → None

##### Borderline examples
- API reference with examples but no guidance → Minimal to Basic (depending on clarity/examples)
- Blog post explaining concept with analogies and one example → Basic
- Tutorial with tasks, checkpoints, and solutions → High
- Product documentation with “Getting Started” and “How-To” flows → Moderate

##### Educational Indicators:
- **Learning objectives**: Clear goals for what reader should learn
- **Skill progression**: Builds from basic to advanced concepts
- **Examples and practice**: Provides concrete examples or exercises
- **Knowledge transfer**: Concepts applicable beyond immediate context

---

#### 11. Reasoning Indicators

**What we're measuring**: Presence and quality of logical reasoning, analysis, and explanatory content.

##### Values & Criteria:

**`analytical`** - Complex reasoning and systematic analysis
- Multi-step arguments with logical progression
- Cause-effect analysis and systematic thinking
- Considers multiple perspectives or variables
- Draws conclusions from evidence and reasoning
- Example: Research analysis, complex problem-solving, systematic evaluations

**`explanatory`** - Clear explanations with logical flow
- Explains how or why things work
- Shows cause-effect relationships clearly
- Educational reasoning that builds understanding
- Logical connections between concepts
- Example: Good tutorials, educational content, how-to explanations

**`basic_reasoning`** - Simple logical connections
- Some logical connections between ideas
- Basic explanations of concepts or processes
- Elementary analytical thinking
- Simple cause-effect relationships
- Example: Basic explanations, simple arguments, elementary analysis

**`minimal`** - Limited reasoning, mostly descriptive
- Primarily describes what rather than why or how
- Few logical connections between ideas
- Mostly factual statements without analysis
- Little explanatory content
- Example: Basic descriptions, simple factual content, minimal analysis

**`none`** - No clear reasoning present
- Purely descriptive content
- Simple factual listing without connections
- Narrative content without analysis
- No logical argumentation or explanation
- Example: Simple lists, basic narratives, pure description

##### Thinking-trace signals (what to look for)
- Stepwise structure: numbered steps in proofs/derivations/solutions; “First… therefore… hence… so…”
- Hypothesis and test: assumptions, intermediate results, counterexamples, sanity checks
- Tool- or method-calls: named algorithms, theorems, lemmas, or procedures invoked and justified
- Error analysis or reflection: “we tried X, failed because Y, so we…”, “limitations,” “edge cases”
- Intermediate artifacts: scratch calculations, partial code reasoning, sub-problems and sub-claims

##### Disambiguation rules
- Explanatory vs Analytical: explanations tell how; analytical shows multi-step inference with evidence and intermediate claims
- Worked example vs Mere answer: worked examples expose steps and justification; mere answers without steps are not reasoning-rich
- Procedural vs Reasoning: procedural lists actions; reasoning links actions via logic, evidence, or constraints

##### Automation heuristics
- Lexical cues: because, therefore, thus, hence, suppose/assume, we conclude, by induction, lemma/theorem/proof, O(n), hypothesis, counterexample
- Structure cues: presence of proof blocks, derivations (e.g., “Proof.”, “QED”, TeX environments), multi-step numeric calculations
- Program reasoning: code comments like “// invariant”, “// complexity”, pre/post-conditions, test reasoning
- Thresholding: count reasoning cues per 1k tokens; with ≥2 structural cues or ≥5 lexical cues → at least explanatory; proofs/derivations → analytical

##### Quick decision tree
- Is there a proof/derivation or multi-step argument with intermediate claims? → analytical
- Else, does it explain why/how with cause-effect and logical links? → explanatory
- Else, are there simple logical connections or one-step justifications? → basic_reasoning
- Else, does it mostly describe without connecting ideas? → minimal/none

##### Borderline examples
- Answer-only solutions (final numeric result without steps) → minimal
- Step-by-step math solution with intermediate equations → analytical
- “How it works” article connecting 23 causal steps without data → explanatory
- Troubleshooting log with attempts and justifications → analytical if causal chain is explicit; otherwise explanatory

##### Key Reasoning Patterns to Identify:
- **Cause-effect**: "Because X, therefore Y"
- **Problem-solution**: Identifies problems and proposes solutions
- **Comparison**: Analyzes similarities and differences
- **Logical progression**: Ideas build on previous ideas
- **Evidence-based conclusions**: Draws conclusions from presented evidence

---

### Audience and Purpose

#### 12. Audience Level

**What we're measuring**: Intended sophistication level and background knowledge assumptions of the target audience.

##### Values & Criteria:

**`expert`** - Highly specialized professional/academic content
- Assumes deep domain expertise and advanced training
- Uses technical terminology without explanation
- Content for practitioners actively working in specialized fields
- Example: Climate modeling methodology in Nature Climate Change, research papers, technical specifications, expert-to-expert communications

**`advanced`** - Educated adult audience with analytical skills
- Assumes higher education and critical thinking ability
- Explains specialized concepts but uses sophisticated language
- Intellectually challenging but accessible to educated generalists
- Example: Complex climate change analysis in The Atlantic, quality journalism, policy analysis, advanced general interest content

**`general`** - General adult audience
- Accessible to most educated adults without specialized background
- Explains technical concepts when introduced
- Uses clear language while maintaining intellectual substance
- Example: Quality journalism, general interest articles, accessible explanations of complex topics

**`beginner`** - Introductory level with minimal prerequisites
- Explains basic concepts and terminology
- Builds up from fundamental principles
- Assumes minimal prior knowledge of the subject area
- Example: Introductory tutorials, beginner guides, basic explanations, getting-started content

**`youth`** - Targeted at teenagers and young adults (ages 13-19)
- Age-appropriate complexity with contemporary cultural references
- Sophisticated enough for developing critical thinking but accessible
- May address topics relevant to adolescent experiences and concerns
- Example: High school educational content, young adult literature, teen-focused explanations, college prep materials

**`children`** - Designed specifically for children
- Simple language and concepts appropriate for young readers
- Educational content designed for elementary/middle school levels
- Age-appropriate topics and complexity
- Example: Children's educational content, elementary school materials, simple explanations for young learners

##### Assessment Guidelines:
- **Professional context**: Is this content designed for workplace use vs. general learning?
- **Terminology density**: How much specialized vocabulary is used without explanation?
- **Concept complexity**: How sophisticated are the ideas and their development?
- **Background assumptions**: What education level and domain knowledge does the author assume?

**Cross-Linguistic Considerations:**
- Expert terminology density varies by language (German allows more compound terms)
- Formality markers differ across cultures
- Educational level assumptions vary by country's education system
- Age-appropriate content differs across cultures

---

#### 13. Commercial Bias

**What we're measuring**: How much commercial interests influence the objectivity and informational value of content.

##### Values & Criteria:

**`none`** - No commercial influence detected  
- Objective, informational presentation
- No promotional language or commercial agenda
- Focus purely on informing or educating
- Example: Academic papers, objective journalism, educational content

**`minimal`** - Slight commercial context but maintains objectivity
- May mention products/services but in informational context
- Maintains balanced, objective tone
- Commercial mentions serve informational purpose
- Example: Product reviews with balanced analysis, informational articles mentioning relevant products

**`moderate`** - Some commercial influence on content
- Mix of informational and promotional content
- Some promotional language but still provides useful information
- Commercial interests somewhat visible but not dominant
- Example: Company blogs with useful information, sponsored content with actual value

**`heavy`** - Strong commercial bias throughout
- Primarily promotional with some informational elements
- Heavy use of marketing language and persuasive techniques
- Clear commercial agenda affects content objectivity
- Example: Marketing articles disguised as information, heavily biased product comparisons

**`pure_marketing`** - Entirely commercial/promotional content
- No genuine informational value beyond promotion
- Pure marketing copy or advertising material
- Designed solely to drive sales or conversions
- Example: Sales pages, pure advertising copy, promotional brochures

##### Key Indicators:
- **Language tone**: Objective vs. promotional language
- **Primary purpose**: Inform vs. persuade/sell
- **Balance**: Are alternatives/drawbacks mentioned?
- **Call-to-action**: Subtle information vs. obvious sales pitch

---

#### 14. Time-Sensitivity

**What we're measuring**: How time-sensitive the content is - whether its value degrades over time or remains stable.

##### Values & Criteria:

**`evergreen`** - Content remains valuable indefinitely
- Fundamental concepts, principles, theories
- Historical information and established facts
- Skills and techniques that don't change
- Reference materials with lasting value
- Example: Mathematical proofs, language grammar guides, classical literature analysis, basic cooking techniques

**`slowly_changing`** - Content remains valuable for years
- Best practices that evolve slowly
- Technical content that updates every few years
- Cultural and social topics with gradual change
- Example: Programming language tutorials, academic textbooks, industry standards, educational curricula

**`regularly_updating`** - Content valuable for months to a year
- Industry trends and market analysis
- Technology reviews and comparisons
- Policy discussions and current research
- Example: Software framework guides, business strategies, product reviews, research summaries

**`time_sensitive`** - Content value degrades quickly
- News and current events
- Time-bound information (prices, schedules, availability)
- Temporary situations or short-term trends
- Real-time data and statistics
- Example: Stock prices, weather reports, breaking news, event announcements, sales/promotions

##### Key Decision Points:
- **Core question**: If someone reads this in 2 years, will it still be valuable?
- **Update frequency**: How often does this type of information typically change?
- **Temporal references**: Does the content heavily reference "now," "recently," "currently"?
- **Subject matter stability**: Is this about unchanging principles or evolving situations?

---

### Safety and Compliance

#### 15. Content Safety

**What we're measuring**: Presence of inappropriate, harmful, or legally problematic content.

##### Values & Criteria:

**`safe`** - Appropriate for all contexts
- No concerning content of any type
- Professional, appropriate language throughout
- Suitable for general audiences including workplace settings

**`mild_concerns`** - Minor issues that don't constitute major problems
- Occasional mild profanity in context
- Brief mentions of sensitive topics handled appropriately  
- Minor concerns that don't affect overall suitability
- Example: Historical discussions of sensitive topics, professional content with mild language

**`nsfw`** - Not safe for work or general audiences
- Explicit sexual content or graphic descriptions
- Adult themes requiring content warnings
- Graphic violence or disturbing imagery descriptions
- Example: Adult content, graphic medical descriptions, explicit violence

**`harmful`** - Potentially harmful content requiring careful handling
- Content promoting dangerous activities or self-harm
- Hate speech targeting individuals or groups
- Violent content glorifying harm to others
- Example: Self-harm content, hate speech, dangerous "how-to" guides

**`illegal`** - Illegal content requiring immediate rejection
- Content promoting clearly illegal activities
- Material that violates laws in major jurisdictions
- Example: Terrorist content, child exploitation

##### Safety Assessment Guidelines:
- **Context matters**: Medical/educational discussions of sensitive topics may be appropriate
- **Intent matters**: Discussing harmful topics for educational purposes vs. promoting them
- **Audience consideration**: Content appropriate for experts may not be safe for general audiences

---

#### 16. PII Presence

**What we're measuring**: Whether the content contains personally identifiable information that could identify private individuals.

##### Values & Criteria:

**`no_pii`** - No personal information detected
- No names of private individuals
- No contact information (emails, phones, addresses)
- No identification numbers
- Public figures and officials mentioned by name are acceptable
- Example: News articles about politicians, technical documentation, general information

**`contains_pii`** - Contains potentially identifiable information
- Names of private individuals (non-public figures)
- Email addresses, phone numbers, physical addresses
- ID numbers (SSN, passport, driver's license, employee IDs)
- Medical information about identifiable individuals
- Financial account information
- Example: Personal blogs with full names, leaked databases, medical case studies with identifying info

##### Key Decision Points:
- **Public vs. Private figures**: Politicians, celebrities, CEOs = public (no PII flag); private citizens = PII
- **Context matters**: Academic paper authors and their institutional emails = typically no PII; personal emails in forums = PII
- **Aggregated vs. Individual**: Statistical data = no PII; individual records = PII

---

### Geographic Relevance

#### 17. Regional Relevance

**What we're measuring**: Primary regional, cultural, or geopolitical sphere(s) that the content relates to, regardless of language used.

**Multi-regional content**: Content can be assigned multiple regional labels if it genuinely spans multiple regions. Choose ALL applicable regions rather than forcing a single primary choice. Always output an array for this property, even if only one region applies.

##### Values & Criteria:

**`european`** - European context (EU and broader Europe)
- Content about European countries, EU policies, or pan-European topics
- European cultural perspectives, social systems, or business practices
- References to European cities, institutions, companies, or regulations
- Includes: EU member states, UK, Switzerland, Norway, Balkans, etc.
- Example: GDPR compliance, European Parliament elections, Schengen area travel, European football leagues

**`north_american`** - North American context
- Content about US, Canada, or Mexico
- North American cultural perspectives, USMCA/NAFTA region topics
- References to North American institutions, companies, or issues
- Example: FDA regulations, Silicon Valley tech, NHL, US constitutional law, Canadian healthcare

**`east_asian`** - East Asian context
- Content about China, Japan, Korea (North/South), Taiwan, Mongolia
- East Asian cultural perspectives, Confucian-influenced societies
- References to East Asian economic models, companies, or social systems
- Example: Gaokao exams, K-pop, Shenzhen tech hub, Japanese work culture, Taiwan semiconductor industry

**`south_asian`** - South Asian context  
- Content about India, Pakistan, Bangladesh, Sri Lanka, Nepal, Bhutan, Afghanistan, Maldives
- South Asian cultural perspectives, subcontinental issues
- References to South Asian institutions, economies, or social structures
- Example: IIT entrance exams, Bollywood, cricket leagues, monsoon impacts, caste system discussions

**`southeast_asian`** - Southeast Asian context
- Content about ASEAN countries (Indonesia, Thailand, Vietnam, Philippines, Malaysia, Singapore, etc.)
- Southeast Asian regional perspectives and economic integration
- References to ASEAN policies, regional companies, or cultural phenomena
- Example: ASEAN economic community, Indonesian elections, Singapore financial sector, Thai tourism

**`middle_eastern`** - Middle Eastern and North African context
- Content about Arab states, Iran, Turkey, Israel, North Africa (MENA region)
- Middle Eastern cultural perspectives, Islamic finance, regional conflicts
- References to Middle Eastern institutions, oil economies, or geopolitics
- Example: Gulf Cooperation Council, OPEC decisions, Middle East peace process, Islamic banking

**`sub_saharan_african`** - Sub-Saharan African context
- Content about African countries south of the Sahara
- African Union topics, sub-Saharan development issues
- References to African institutions, economies, or cultural topics
- Example: M-Pesa mobile banking, African Union policies, safari tourism, ubuntu philosophy

**`latin_american`** - Latin American context
- Content about Central and South America, Caribbean
- Latin American cultural perspectives, regional integration (Mercosur, etc.)
- References to Latin American institutions, economies, or social movements
- Example: Mercosur trade, telenovelas, Amazon rainforest, Latin American revolutions

**`oceanian`** - Oceanian context
- Content about Australia, New Zealand, Pacific Island nations
- Oceanian perspectives, Pacific regional issues
- References to Oceanian institutions, companies, or cultural topics
- Example: ANZAC relations, Pacific Island climate change, Australian mining, Māori culture

**`central_asian`** - Central Asian context
- Content about Kazakhstan, Uzbekistan, Turkmenistan, Tajikistan, Kyrgyzstan
- Central Asian perspectives, post-Soviet regional dynamics
- Silk Road region, resource economies, nomadic heritage
- Example: Silk Road initiatives, Caspian Sea resources, post-Soviet transitions

**`russian_sphere`** - Russian/Post-Soviet context
- Content about Russia, Belarus, and strong Russian influence areas
- Post-Soviet perspectives, CIS (Commonwealth of Independent States) topics
- Russian language content about regional (not global) topics
- Example: Russian federal politics, CIS integration, post-Soviet economic transitions

**`global`** - Genuinely international or universal
- Content with truly global scope or application
- International organizations, worldwide phenomena, global comparisons
- Topics that transcend regional boundaries
- Example: UN reports, climate change (global perspective), international standards, pandemic response

**`culturally_neutral`** - No clear regional focus
- Abstract, theoretical, or technical content without regional markers
- Universal scientific, mathematical, or philosophical content
- Content that could apply equally anywhere without modification
- Example: Mathematical proofs, chemical formulas, abstract philosophy, programming concepts

**`indeterminate`** - Cannot determine regional relevance
- Insufficient content to identify regional focus
- Mixed or contradictory regional signals
- Fragment or corrupted content lacking regional context
- Example: Technical specifications without context, isolated data tables

##### Multi-Regional Examples:
- **EU-China trade relations** → `["european", "east_asian"]`
- **NAFTA/USMCA impact on Mexican agriculture** → `["north_american", "latin_american"]`
- **Indian diaspora in the Gulf states** → `["south_asian", "middle_eastern"]`
- **Comparative study of healthcare systems globally** → `["global"]`

##### Regional Identification Guidelines:

**Primary indicators:**
- **Geographic references**: Countries, cities, regions, landmarks mentioned
- **Institutional references**: Governments, companies, universities, organizations specific to region
- **Cultural markers**: Holidays, customs, cultural phenomena, sports, entertainment
- **Political/economic systems**: References to regional political structures, economic blocs
- **Legal/regulatory frameworks**: Region-specific laws, regulations, standards
- **Language context**: While not determinative, language can provide regional hints

**Important distinctions:**
- **Language ≠ Region**: Spanish content about Asian markets = `["east_asian"]`, not `["latin_american"]`
- **Company origin vs. topic**: Apple (US company) operating in India = consider actual content focus
- **Historical vs. current**: Historical content about ancient Rome = `["european"]` if discussing modern implications
- **Diaspora content**: Content about diaspora communities should include both origin and current regions

**Quality checks:**
- If content is in a non-English language but discusses global topics → still mark as `["global"]`
- If content compares multiple regions → mark all regions discussed substantially
- If content is about a specific place but has universal applications → consider both regional and global tags
---

#### 18. Country Relevance

**What we're measuring**: Which specific country or countries (if any) the content is relevant to, globally.

**Note**: Always output an array of country names for this property (even when only a single country applies). Use standard country names from any region worldwide (e.g., "germany", "france", "united_states", "united_kingdom", "china", "japan", "brazil", "india", "south_africa", "australia", "canada", etc.). The array may also contain the special values `supranational` or `none`.

##### Values & Criteria:

**`{COUNTRY_NAME}`** - Content specifically relevant to a single country
- Content explicitly about that country's politics, culture, institutions, or regulations
- Content written from that country's cultural perspective
- Content addressing that country's specific issues, regulations, or cultural phenomena
- Content about that country's cities, companies, institutions, or country-specific topics
- Example: For "germany" → German election coverage, Bundesliga content, German legal analysis
- Example: For "united_states" → US election coverage, NFL content, US legal analysis
- Example: For "japan" → Japanese politics, J-League content, Japanese cultural analysis
- Only use country names listed in ISO-3166. Use "united_kingdom" instead of "england", "wales", etc.

**`supranational`** - For content focused on supranational entities or regions
- International organizations, regional blocs, global institutions
- Content about supranational policies, international organizations, global governance
- Pan-regional analysis that transcends individual countries
- Multi-continental or global institutional content
- Example: UN resolutions, NATO discussions, EU policy analysis, ASEAN agreements, WTO trade rules

**`none`** - For content not specifically relevant to any country
- Abstract, theoretical, or universal content without geographic specificity
- Technical/scientific content that applies globally without country focus
- Content that doesn't reference specific countries, cultures, or national contexts
- Example: Mathematical proofs, universal scientific principles, abstract philosophical discussions


##### Country Identification Criteria:
- **Political content**: Elections, government policies, political parties, political figures specific to the country
- **Cultural content**: National traditions, cultural phenomena, historical events specific to the country
- **Institutional references**: Government bodies, national companies, universities specific to the country
- **Geographic focus**: Cities, regions, landmarks within the country as primary subjects
- **Legal/regulatory**: Laws, regulations, legal frameworks specific to the country
- **Economic content**: National economic policies, country-specific market analysis
- **Sports/media**: National sports leagues, national teams, country-specific media outlets
- **Social issues**: Social policies, demographic topics, social movements specific to the country

---