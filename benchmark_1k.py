"""
ChainLink 1000-Memory Accuracy Benchmark
==========================================
Stress-tests chain reasoning against a realistic 1000-memory dataset.
This simulates a real vibe coder's AI assistant that has accumulated
months of context about a user.

Measures:
- Chain Recall: % of expected chain connections actually found
- Chain Precision: % of flagged chains that are real chains
- Direct Recall: What plain vector search would find (baseline)
- Noise Resistance: Can it find chains buried in 1000 memories?
"""

import json
import time
import os
import sys
import random

random.seed(42)  # Reproducible

# ============================================================
# THE 1000-MEMORY DATASET
# ============================================================
# Organized into clusters with intentional cross-cluster chains.
# Chain connections are BETWEEN clusters — that's the hard part.

MEMORIES = []

# --- HEALTH & MEDICAL (memories 0-59) ---
HEALTH = [
    # Core health facts (0-9)
    "I have a severe shellfish allergy — anaphylaxis risk",                         # 0
    "My doctor prescribed an EpiPen I carry everywhere",                            # 1
    "I take 20mg of Lisinopril daily for blood pressure",                           # 2
    "I'm lactose intolerant but can handle aged cheeses",                           # 3
    "My resting heart rate is around 52 bpm",                                       # 4
    "I have mild asthma — carry a rescue inhaler",                                  # 5
    "Blood type is O negative — universal donor",                                   # 6
    "I'm allergic to penicillin — use azithromycin instead",                        # 7
    "Last colonoscopy was 2024, next one due 2034",                                 # 8
    "I wear prescription glasses — -3.5 in both eyes",                              # 9
    # Exercise (10-19)
    "I run 5K three times a week, usually before 7am",                              # 10
    "My half marathon PR is 1:47:22 set in October 2025",                           # 11
    "I do strength training on Tuesdays and Thursdays",                             # 12
    "I pulled my right hamstring last month — still recovering",                    # 13
    "My gym is Equinox on Market Street — open 5am to 11pm",                       # 14
    "I swim 1500m every Saturday at the YMCA pool",                                 # 15
    "Yoga class on Sunday mornings at CorePower",                                   # 16
    "My Garmin watch tracks all my workouts automatically",                         # 17
    "Average weekly running mileage is about 25 miles",                             # 18
    "I foam roll after every run to prevent injuries",                              # 19
    # Mental health (20-29)
    "I see a therapist every other Wednesday at 2pm",                               # 20
    "I practice meditation for 10 minutes each morning",                            # 21
    "I take 200mg of magnesium glycinate before bed for sleep",                     # 22
    "I use the Calm app for guided sleep meditations",                              # 23
    "Journaling in Obsidian helps me process work stress",                          # 24
    "I do a digital detox every Sunday — no screens after 6pm",                     # 25
    "My therapist recommended the book 'The Body Keeps the Score'",                 # 26
    "I get seasonal affective disorder in winter — use a SAD lamp",                 # 27
    "Breathing exercises help with my presentation anxiety",                        # 28
    "I try to get 7-8 hours of sleep every night",                                  # 29
    # Diet & nutrition (30-39)
    "I'm trying to eat more plant-based meals during the week",                     # 30
    "I drink 3-4 cups of coffee daily — usually cold brew",                         # 31
    "I take vitamin D3 5000 IU daily since my levels were low",                     # 32
    "I meal prep on Sundays for the work week",                                     # 33
    "I'm trying to reduce my sodium intake per doctor's advice",                    # 34
    "Favorite protein shake: whey protein with banana and peanut butter",           # 35
    "I've been intermittent fasting — 16:8 pattern",                                # 36
    "My daily calorie target is around 2400",                                       # 37
    "I track macros using the MyFitnessPal app",                                    # 38
    "I avoid artificial sweeteners — they trigger my migraines",                    # 39
    # Medical history (40-49)
    "I had my appendix removed in 2018",                                            # 40
    "I broke my left wrist skateboarding when I was 14",                            # 41
    "Family history of heart disease — dad had a stent at 58",                      # 42
    "I had COVID in January 2024 — mild symptoms",                                 # 43
    "My cholesterol was borderline high at last checkup — 215 total",              # 44
    "I get migraines about twice a month, usually stress-triggered",                # 45
    "My dentist appointment is next Tuesday at 3pm",                                # 46
    "I had LASIK consultation but decided against it for now",                      # 47
    "I donate blood every 8 weeks at the Red Cross",                                # 48
    "My primary care doctor is Dr. Patel at Stanford Medical",                      # 49
    # Supplements & meds details (50-59)
    "Glucosamine supplements can come from shellfish shells",                       # 50
    "My gym buddy recommended glucosamine for my knee pain",                        # 51
    "I take fish oil 1000mg daily for omega-3s",                                    # 52
    "Creatine monohydrate 5g daily for strength training",                          # 53
    "My pharmacist said Lisinopril can cause a dry cough",                          # 54
    "I use Tiger Balm for muscle soreness after heavy lifts",                       # 55
    "Melatonin 3mg as backup when magnesium doesn't help sleep",                    # 56
    "I switched from ibuprofen to acetaminophen — easier on stomach",               # 57
    "My prescription sunglasses are ready for pickup at LensCrafters",              # 58
    "Annual physical is scheduled for April 10th",                                  # 59
]

# --- FOOD & COOKING (memories 60-129) ---
FOOD = [
    # Cuisines (60-74)
    "I love Thai food, especially green curry and pad thai",                         # 60
    "Thai green curry paste typically contains shrimp paste (kapi)",                 # 61
    "My favorite restaurant is Siam Garden on 5th Street",                          # 62
    "I've been learning to make homemade pasta on weekends",                        # 63
    "Parmesan cheese is naturally lactose-free due to aging process",               # 64
    "I discovered a great ramen shop in Japantown — Marufuku",                     # 65
    "I love making sourdough bread — my starter is 2 years old",                    # 66
    "My go-to weeknight dinner is stir-fry with whatever's in the fridge",         # 67
    "I prefer extra-spicy — always ask for Thai hot at restaurants",                 # 68
    "Best pizza in the city is at Tony's Slice Shop in North Beach",               # 69
    "I make a mean shakshuka for weekend brunch",                                   # 70
    "Korean BBQ at KBBQ Plus is our go-to for group dinners",                      # 71
    "I've been experimenting with fermentation — kimchi and kombucha",              # 72
    "My sourdough discard pancakes are a family favorite",                          # 73
    "I roast my own coffee beans — currently on Ethiopian Yirgacheffe",             # 74
    # Cooking skills (75-84)
    "I just got a sous vide machine — experimenting with steaks",                   # 75
    "My cast iron skillet is my most-used kitchen tool",                            # 76
    "I make homemade chicken stock every month from scraps",                        # 77
    "I learned to make fresh mozzarella — it's surprisingly easy",                  # 78
    "My sourdough bread won second place at the county fair",                       # 79
    "I cure my own bacon — takes about a week",                                     # 80
    "I just started growing herbs on my kitchen windowsill",                        # 81
    "My pasta machine is a Marcato Atlas 150",                                      # 82
    "I make my own hot sauce from garden habaneros",                                # 83
    "I've been practicing knife skills — can julienne pretty fast now",             # 84
    # Grocery & meal planning (85-94)
    "Weekly Costco run is usually Saturday morning",                                 # 85
    "I get a CSA box from Full Belly Farm every two weeks",                         # 86
    "I order specialty spices from Burlap & Barrel online",                         # 87
    "We spend about $800/month on groceries for the family",                        # 88
    "I keep a running grocery list in the Apple Reminders app",                     # 89
    "Trader Joe's frozen meals are my backup for busy nights",                      # 90
    "I buy whole chickens and break them down myself — cheaper",                    # 91
    "Our chest freezer in the garage is almost full",                               # 92
    "I buy coffee beans from Blue Bottle subscription — 2 bags/month",             # 93
    "I prefer organic produce when available but not militant about it",            # 94
    # Dietary specifics (95-104)
    "I can't eat raw onions — they give me terrible heartburn",                     # 95
    "I love sushi but always check for shellfish cross-contamination",              # 96
    "My wife is vegetarian — I cook meat separately",                               # 97
    "Emma will only eat chicken nuggets and mac and cheese",                        # 98
    "I've been trying to eat more fermented foods for gut health",                  # 99
    "I avoid high-mercury fish — no swordfish or king mackerel",                    # 100
    "Dark chocolate 85%+ is my go-to dessert",                                     # 101
    "I drink about 2 liters of water daily — have a Hydro Flask",                  # 102
    "Green tea in the afternoon instead of coffee to avoid sleep issues",           # 103
    "I occasionally do a bone broth fast when I feel run down",                     # 104
    # Food memories & events (105-114)
    "The best meal I ever had was at Noma in Copenhagen in 2022",                   # 105
    "Our wedding cake was carrot cake from Miette bakery",                          # 106
    "I hosted a paella party for 20 people last summer — huge hit",                # 107
    "My grandmother's lasagna recipe is a family treasure",                         # 108
    "I ate crickets in Oaxaca — surprisingly nutty and good",                       # 109
    "Our anniversary dinner is always at Gary Danko",                               # 110
    "I tried durian in Singapore — never again",                                    # 111
    "The farmers market on Saturday has the best stone fruit",                       # 112
    "I won a chili cookoff at work with my smoked brisket chili",                   # 113
    "My mom's pho recipe takes 12 hours but it's worth it",                        # 114
    # Kitchen equipment (115-129)
    "My KitchenAid mixer is the workhorse of my kitchen",                          # 115
    "I just upgraded to a Breville Barista Express espresso machine",               # 116
    "My Instant Pot is great for quick weeknight meals",                            # 117
    "I have a Big Green Egg smoker in the backyard",                               # 118
    "My knife set is Wüsthof Classic — the 8-inch chef's knife is my favorite",   # 119
    "I use a Vitamix blender for smoothies and soups",                              # 120
    "My kitchen scale is essential for baking — I weigh everything",               # 121
    "I ferment in Le Parfait jars — they're airtight",                             # 122
    "I just got a dehydrator — making beef jerky this weekend",                    # 123
    "My pizza stone cracked — need to replace it with a steel",                    # 124
    "Our outdoor kitchen setup has a gas grill and smoker",                         # 125
    "I use ChefSteps Joule for sous vide — app controlled",                        # 126
    "My food processor is a Cuisinart — use it mainly for pie dough",             # 127
    "I have a carbon steel wok — essential for proper stir-fry",                   # 128
    "My bread proofing basket (banneton) needs replacing",                          # 129
]

# --- TRAVEL (memories 130-209) ---
TRAVEL = [
    # Upcoming Japan trip (130-149)
    "I'm flying to Tokyo next month for a two-week trip",                           # 130
    "Japan is 14 hours ahead of US Pacific time",                                   # 131
    "I get bad jet lag going eastbound — takes me 3-4 days to adjust",             # 132
    "I always book aisle seats on flights over 4 hours",                            # 133
    "My passport expires in September 2026",                                        # 134
    "Japan requires 6 months passport validity for entry",                          # 135
    "I have TSA PreCheck — Known Traveler Number 12345678",                        # 136
    "Japanese convenience stores have amazing onigiri and bento",                   # 137
    "I want to visit the Tsukiji outer market for fresh sushi",                     # 138
    "I booked an Airbnb in Shibuya for the first week",                            # 139
    "Second week in Kyoto — staying near Gion district",                           # 140
    "I want to take the bullet train (Shinkansen) from Tokyo to Kyoto",            # 141
    "Japan Rail Pass is worth it for a 2-week trip",                                # 142
    "I need to exchange some USD to yen — Japan is still cash-heavy",              # 143
    "I want to visit TeamLab Borderless in Azabudai",                              # 144
    "My SIM card plan covers Japan — T-Mobile international",                      # 145
    "I want to hike part of the Kumano Kodo pilgrimage trail",                     # 146
    "Spring cherry blossom season in Japan is late March to mid April",            # 147
    "I need to learn some basic Japanese phrases before the trip",                  # 148
    "Japanese etiquette: bow when greeting, remove shoes indoors",                  # 149
    # Past travel (150-169)
    "We went to Italy for our honeymoon in 2019 — Rome, Florence, Amalfi",        # 150
    "Best snorkeling I've done was in Belize — Hol Chan Marine Reserve",          # 151
    "I hiked the Inca Trail to Machu Picchu in 2021",                              # 152
    "We did a road trip through Iceland — Ring Road in 10 days",                   # 153
    "I've been to 23 countries so far — goal is 50 by age 50",                    # 154
    "Paris in December was magical — loved the Christmas markets",                  # 155
    "I got food poisoning in Mexico City — worst travel experience",               # 156
    "Our family trip to Yellowstone was amazing — saw wolves",                     # 157
    "I climbed Kilimanjaro in 2020 — hardest thing I've ever done",               # 158
    "Favorite city in Europe is Barcelona — the architecture is incredible",       # 159
    "I've done the Camino de Santiago — 500 miles in 33 days",                     # 160
    "We went whale watching in Monterey Bay — saw humpbacks",                      # 161
    "I visited the DMZ in South Korea — eerie and fascinating",                    # 162
    "Safari in Tanzania — saw the Big Five in Serengeti",                           # 163
    "New Zealand's Milford Sound was breathtaking",                                # 164
    "I tried bungee jumping in Queenstown — terrifying and amazing",               # 165
    "Vietnam's Ha Long Bay was one of the most beautiful places I've seen",        # 166
    "I did a cooking class in Chiang Mai — learned to make khao soi",             # 167
    "The Northern Lights in Tromsø Norway were unforgettable",                     # 168
    "I got my PADI open water certification in Thailand",                           # 169
    # Travel preferences (170-189)
    "I prefer boutique hotels over big chains",                                     # 170
    "I always buy travel insurance for international trips",                        # 171
    "I use Google Maps offline downloads for navigation abroad",                    # 172
    "I pack carry-on only for trips under 2 weeks",                                # 173
    "My go-to travel backpack is an Osprey Farpoint 40",                           # 174
    "I always bring a packable rain jacket — Columbia brand",                      # 175
    "I use a VPN when traveling — ExpressVPN",                                      # 176
    "I keep a copy of my passport in my email as backup",                           # 177
    "I always arrive at the airport 3 hours early for international",              # 178
    "I prefer morning flights — less likely to be delayed",                         # 179
    "I use Packing Pro app to make sure I don't forget anything",                  # 180
    "I bring a universal power adapter for international trips",                    # 181
    "I always get local SIM or eSIM instead of roaming",                           # 182
    "I prefer window seats for short flights, aisle for long ones",                # 183
    "I use Priority Pass for airport lounge access",                                # 184
    "I take melatonin to help adjust to new time zones",                           # 185
    "I always research local scams before visiting a new country",                 # 186
    "I keep an emergency credit card separate from my wallet",                      # 187
    "I use Wise (TransferWise) for foreign currency exchanges",                     # 188
    "I download offline language packs in Google Translate",                        # 189
    # Travel planning (190-209)
    "We're thinking about Costa Rica for winter break next year",                   # 190
    "I want to do a cycling tour in Provence, France",                              # 191
    "Bucket list: see the aurora borealis from a glass igloo in Finland",           # 192
    "I'm saving frequent flyer miles on United Airlines",                           # 193
    "I have Chase Sapphire Reserve — good for travel points",                      # 194
    "Our family loves national parks — America the Beautiful pass",                # 195
    "I want to visit Patagonia for hiking and glaciers",                            # 196
    "We discussed doing a Mediterranean cruise next year",                          # 197
    "I need to renew my Global Entry — expires in 2027",                           # 198
    "Thinking about a ski trip to Whistler in February",                            # 199
    "I want to take the Trans-Siberian Railway someday",                            # 200
    "We might visit my wife's family in Chennai, India this summer",               # 201
    "I'm interested in a safari lodge in Botswana — Okavango Delta",              # 202
    "I want to see the terracotta warriors in Xi'an, China",                       # 203
    "We want to take Emma to Disney World when she's 8",                           # 204
    "I'm researching overwater bungalows in the Maldives",                         # 205
    "We want to visit the Christmas markets in Vienna and Prague",                  # 206
    "I'm interested in a food tour in Bologna, Italy",                              # 207
    "I want to drive the Amalfi Coast road next time we're in Italy",              # 208
    "Camping trip to Joshua Tree planned for next month",                           # 209
]

# --- WORK (memories 210-309) ---
WORK = [
    # Job basics (210-224)
    "I'm a senior software engineer at Acme Corp",                                  # 210
    "I've been at Acme for 3 years — joined as a mid-level engineer",              # 211
    "My manager is Sarah Chen — she's been great",                                  # 212
    "Our team has 8 engineers, 2 designers, and a PM",                             # 213
    "I work remotely from San Francisco — company is in NYC",                      # 214
    "My salary is $195k base with up to 15% bonus",                                # 215
    "I have 20 days PTO plus company holidays",                                     # 216
    "Our tech stack is Python, TypeScript, PostgreSQL, and AWS",                   # 217
    "I sit on the platform team — we build internal tools",                        # 218
    "My skip-level manager is VP of Engineering, David Park",                       # 219
    "I'm on the on-call rotation — every 4th week",                                # 220
    "Our sprint cycles are 2 weeks — planning on Mondays",                         # 221
    "I use Slack, Jira, and Confluence for work communication",                     # 222
    "Our team standup is at 9am Pacific every weekday",                            # 223
    "I prefer async communication over meetings when possible",                    # 224
    # Current project (225-244)
    "I'm leading the migration from MongoDB to PostgreSQL",                        # 225
    "The database migration deadline is April 15th",                                # 226
    "Our production database has 2.3TB of data to migrate",                         # 227
    "We're using pgloader for the actual data migration",                           # 228
    "I wrote a custom schema mapping tool in Python",                               # 229
    "The migration affects 12 microservices that need updating",                   # 230
    "We identified 47 MongoDB-specific queries that need rewriting",               # 231
    "Performance benchmarks show PostgreSQL is 30% faster for our workload",       # 232
    "We set up a shadow read system to compare MongoDB vs PostgreSQL results",     # 233
    "The migration has a rollback plan if things go wrong",                         # 234
    "Sarah wants weekly progress reports on Fridays by 4pm",                       # 235
    "We're in phase 2 of 4 — currently migrating the user service",               # 236
    "I need to update the API documentation after migration",                       # 237
    "We discovered 3 data consistency issues during shadow reads",                 # 238
    "The staging environment migration is complete and passing tests",             # 239
    "We're planning a load test with 10x normal traffic before go-live",           # 240
    "I set up Datadog dashboards to monitor migration health",                     # 241
    "The estimated cost savings from migration is $4k/month",                      # 242
    "We need to coordinate the cutover with the mobile team",                      # 243
    "I'm writing a post-mortem template for after the migration",                  # 244
    # Career & skills (245-259)
    "I'm studying for the AWS Solutions Architect certification",                   # 245
    "I want to move into a staff engineer role in the next 2 years",              # 246
    "I gave a tech talk on database migration patterns last month",                # 247
    "I mentor two junior engineers on the team",                                    # 248
    "I'm learning Rust on the side — might be useful for performance work",        # 249
    "I contribute to open source — mainly Python libraries",                       # 250
    "My strongest skills are backend systems and database optimization",           # 251
    "I'm weak on frontend — want to improve my React skills",                     # 252
    "I read 'Designing Data-Intensive Applications' — changed how I think",       # 253
    "I attend the local Python meetup every first Thursday",                        # 254
    "I've been writing a blog about engineering leadership",                        # 255
    "I did a presentation at PyCon 2025 on async Python patterns",                 # 256
    "I'm interested in machine learning but haven't found time to dive in",        # 257
    "I pair program with teammates at least twice a week",                          # 258
    "I use Todoist for personal task management",                                   # 259
    # Work culture & processes (260-274)
    "We do code reviews — every PR needs at least 2 approvals",                    # 260
    "Our CI/CD pipeline uses GitHub Actions",                                       # 261
    "We follow trunk-based development with feature flags",                         # 262
    "Our error budget for the quarter is 99.9% uptime",                            # 263
    "We use Terraform for infrastructure as code",                                  # 264
    "I wrote our team's incident response runbook",                                 # 265
    "We do blameless post-mortems after every incident",                            # 266
    "Our team does a retrospective every other Friday",                            # 267
    "We use feature flags via LaunchDarkly for gradual rollouts",                  # 268
    "I set up automated canary deployments for our services",                       # 269
    "We have a weekly architecture review on Wednesdays at 2pm",                   # 270
    "Our monitoring stack is Datadog + PagerDuty",                                  # 271
    "We do quarterly OKRs — I own the migration OKR this quarter",                # 272
    "I maintain our team's technical decision records (ADRs)",                      # 273
    "We use semantic versioning for all our internal packages",                     # 274
    # Work relationships (275-289)
    "I have a great working relationship with the data science team",              # 275
    "Our PM Alex is really good at shielding us from scope creep",                # 276
    "The design team lead, Maya, is easy to collaborate with",                     # 277
    "I'm working with the security team on SOC 2 compliance",                      # 278
    "Our CTO, James, does a monthly all-hands AMA",                                # 279
    "I grab lunch with my work buddy Tom almost every day",                         # 280
    "The DevOps team lead, Priya, helped set up our CI pipeline",                  # 281
    "I'm on the hiring committee — we have 3 open positions",                     # 282
    "Our team's Slack channel has too many notifications",                          # 283
    "I have a 1:1 with Sarah every Tuesday at 10am",                               # 284
    "The QA team catches about 15% of bugs our tests miss",                        # 285
    "I coordinate with the frontend team on API contracts",                         # 286
    "Our offshore team in Bangalore handles weekend on-call",                      # 287
    "I'm helping recruit a database specialist for the team",                       # 288
    "The sales team keeps requesting custom API endpoints",                        # 289
    # Work tech (290-309)
    "I use VS Code with the Vim extension for all coding",                          # 290
    "My preferred stack is Python + FastAPI + PostgreSQL",                          # 291
    "I run Arch Linux on my personal laptop",                                       # 292
    "I have a Claude Code subscription and use it daily for work",                 # 293
    "My work laptop is a MacBook Pro M3 Max with 64GB RAM",                        # 294
    "I use Docker Desktop for local development",                                   # 295
    "I have 3 monitors at my home office setup",                                    # 296
    "I use tmux for terminal multiplexing",                                         # 297
    "My dotfiles are version controlled on GitHub",                                 # 298
    "I use 1Password for all password management",                                  # 299
    "I have a standing desk — alternate sitting/standing hourly",                  # 300
    "My mechanical keyboard is a ZSA Moonlander — split ergonomic",              # 301
    "I use Raycast as my Mac launcher — replaced Alfred",                          # 302
    "I keep my terminal prompt minimal — just path and git branch",                # 303
    "I use GitHub Copilot alongside Claude Code",                                   # 304
    "My home office internet is Sonic fiber — 1Gbps symmetric",                    # 305
    "I use Tailscale for VPN between my devices",                                   # 306
    "I back up everything to Backblaze B2",                                         # 307
    "My IDE font is JetBrains Mono at 14px",                                       # 308
    "I use Bear app for quick work notes — syncs across devices",                  # 309
]

# --- FAMILY & SOCIAL (memories 310-409) ---
FAMILY = [
    # Wife (310-324)
    "My wife's name is Priya — we've been married 8 years",                        # 310
    "Priya is a product manager at Google",                                         # 311
    "Priya's birthday is March 28th",                                               # 312
    "Priya is vegetarian — has been since college",                                 # 313
    "Priya's parents live in Chennai, India",                                       # 314
    "Priya loves hiking — we try to do a hike every weekend",                      # 315
    "Priya is training for her first 10K race",                                     # 316
    "Priya's favorite flowers are peonies",                                         # 317
    "We met at a mutual friend's dinner party in 2016",                             # 318
    "Our anniversary is June 15th",                                                 # 319
    "Priya wants us to take a pottery class together",                              # 320
    "Priya is learning to play ukulele",                                            # 321
    "Priya's favorite show is 'The Great British Bake Off'",                       # 322
    "We have a weekly date night on Fridays — alternate who plans",                # 323
    "Priya doesn't drink coffee — only chai",                                      # 324
    # Daughter Emma (325-344)
    "My daughter Emma turns 7 in April",                                            # 325
    "Emma is obsessed with dinosaurs, especially T-Rex",                            # 326
    "Emma just started taking swim lessons at the YMCA",                            # 327
    "Emma's best friend is Maya from school",                                       # 328
    "Emma is in first grade at Presidio Elementary",                                # 329
    "Emma loves drawing — goes through a sketchbook a month",                      # 330
    "Emma wants a kitten for her birthday",                                         # 331
    "Emma is afraid of thunderstorms",                                              # 332
    "Emma's favorite food is mac and cheese — the Annie's brand",                  # 333
    "Emma has a stuffed dinosaur named Rex she sleeps with every night",           # 334
    "Emma's soccer practice is Saturday mornings at 10am",                          # 335
    "Emma can read chapter books now — currently on 'Diary of a Wimpy Kid'",      # 336
    "Emma wants to be a paleontologist when she grows up",                          # 337
    "Emma's school talent show is in May — she's doing a dance",                   # 338
    "I volunteer at Emma's school library every other Thursday",                    # 339
    "Emma gets carsick on long drives — we keep ginger candy in the car",          # 340
    "Emma loves the California Academy of Sciences — we have a membership",        # 341
    "Emma's bedtime is 8pm on school nights, 9pm on weekends",                     # 342
    "Emma is learning to ride a bike without training wheels",                      # 343
    "We're planning Emma's birthday party at the Natural History Museum",           # 344
    # Dog Max (345-354)
    "We adopted a golden retriever named Max last year",                            # 345
    "Max has a chicken allergy — eats grain-free fish-based food",                 # 346
    "Max goes to doggy daycare on Tuesdays and Thursdays",                         # 347
    "Max's vet is Dr. Kim at SF Vet Specialists",                                  # 348
    "Max weighs about 70 pounds — needs to lose 5",                                # 349
    "Max loves the beach — we take him to Fort Funston on weekends",              # 350
    "Max is trained but still pulls on the leash sometimes",                        # 351
    "Max has a Kong toy that keeps him busy for hours",                             # 352
    "Max is afraid of the vacuum cleaner",                                          # 353
    "Max's annual checkup is next month",                                           # 354
    # Extended family (355-374)
    "My parents live in Portland, Oregon",                                          # 355
    "My dad retired from teaching high school physics last year",                   # 356
    "My mom is a retired nurse — she's active in community gardening",             # 357
    "My parents visit every Thanksgiving — they drive down from Portland",         # 358
    "My brother Raj lives in Seattle — he's a dentist",                            # 359
    "Raj has two kids — Anaya (10) and Dev (8)",                                   # 360
    "We do a family vacation together every summer",                                # 361
    "My mom's birthday is in November — she loves gardening books",                # 362
    "My dad has mild hearing loss — we got him hearing aids",                       # 363
    "My parents' 40th anniversary is next year",                                    # 364
    "Priya's sister Anita lives in London — she's visiting in July",              # 365
    "Priya's parents come for a month every winter",                                # 366
    "My grandmother in India turned 90 last year",                                  # 367
    "My uncle Kumar is a cardiologist in Houston",                                  # 368
    "Family group chat is on WhatsApp — 23 people",                                # 369
    "We host Diwali dinner for about 30 people every year",                        # 370
    "My cousin's wedding is in Mumbai in December",                                 # 371
    "My dad and I bond over cricket — we watch IPL together",                      # 372
    "My mom sends us care packages with Indian snacks monthly",                    # 373
    "My brother and I play online chess every Sunday evening",                      # 374
    # Friends (375-394)
    "My best friend since college is Jason — he's a lawyer in LA",                 # 375
    "I play pickup basketball on Wednesday evenings with friends",                  # 376
    "My neighbor Dave and I share a fence — he's a great guy",                    # 377
    "Our friends group does a camping trip every Labor Day",                        # 378
    "My running buddy is Carlos — we do the Saturday long run together",           # 379
    "I'm in a book club that meets the last Tuesday of every month",               # 380
    "My friend Lisa is a sommelier — she picks wine for our dinners",             # 381
    "I play poker with a group of 6 every other Friday",                           # 382
    "My college roommate Mike just had his first kid",                              # 383
    "I volunteer at the local food bank on the first Saturday of each month",      # 384
    "Our neighbors Sarah and Tom host a block party every July 4th",              # 385
    "I mentor a high school student through the FIRST robotics program",          # 386
    "My friend Andrea is a personal trainer — she designs my workouts",           # 387
    "I'm in a fantasy football league with 12 friends",                            # 388
    "My friend Kevin is a photographer — he did our family photos",               # 389
    "We have a couples dinner group — 4 families, monthly rotation",              # 390
    "My surfing buddy Jake and I go to Pacifica on good swell days",              # 391
    "I play guitar in a jazz combo with 4 friends — we rehearse Mondays",         # 392
    "My friend Tom (not neighbor Tom) works at OpenAI",                            # 393
    "Our friend group has an annual fantasy draft party with BBQ",                 # 394
    # Home & neighborhood (395-409)
    "We live in the Sunset District of San Francisco",                              # 395
    "Our house is a 3-bedroom Victorian built in 1928",                            # 396
    "We have a small backyard with a raised garden bed",                            # 397
    "Our mortgage is $4,200/month — 30-year fixed at 3.2%",                       # 398
    "The house needs a new roof in the next 2-3 years — estimate $25k",           # 399
    "We installed solar panels last year — saves about $150/month",                # 400
    "Our garage is half workshop, half storage",                                    # 401
    "The neighborhood is quiet and family-friendly",                                # 402
    "We're 10 minutes from Golden Gate Park by bike",                              # 403
    "Our property tax is about $12k/year",                                          # 404
    "We need to replace the hot water heater — it's 15 years old",               # 405
    "I built a pergola in the backyard last summer",                                # 406
    "Our kitchen renovation is budgeted for next year — about $50k",              # 407
    "The neighbor's tree drops leaves in our yard constantly",                      # 408
    "We have earthquake insurance — $3k deductible",                               # 409
]

# --- FINANCE (memories 410-479) ---
FINANCE = [
    # Retirement & investments (410-429)
    "I max out my 401k contribution every year — $23,000 for 2025",                # 410
    "My 401k balance is about $285k — mostly in index funds",                      # 411
    "I have a Roth IRA with $72k at Vanguard",                                     # 412
    "I invest in VTSAX, VTIAX, and VBTLX — three-fund portfolio",                # 413
    "I rebalance my portfolio quarterly",                                           # 414
    "I have about $15k in an emergency fund at Ally Bank",                          # 415
    "I'm saving for a down payment on a rental property — goal $80k",             # 416
    "Monthly savings rate is about 25% of take-home pay",                          # 417
    "I use Personal Capital to track net worth — updated monthly",                 # 418
    "I have $8k in a 529 plan for Emma's college",                                 # 419
    "I contribute $500/month to Emma's 529",                                        # 420
    "Our household income is about $380k combined",                                 # 421
    "I have RSUs at Acme that vest quarterly — about $30k/year",                  # 422
    "I sold some Acme stock last year and owe capital gains tax",                   # 423
    "I use tax-loss harvesting in my taxable brokerage account",                   # 424
    "My financial advisor is at Wealthfront — robo-advisor",                       # 425
    "I contribute to an HSA — $3,850 max for 2025",                                # 426
    "I keep 6 months expenses in emergency fund — about $30k target",              # 427
    "I buy I-bonds when the rate is above 5%",                                      # 428
    "Estimated tax payments due quarterly — I use TurboTax",                       # 429
    # Debt & expenses (430-449)
    "I have $42k in student loans at 5.2% interest",                                # 430
    "Student loan payment is $450/month on extended repayment",                     # 431
    "Monthly mortgage budget for rental property is around $3,200",                 # 432
    "We have no credit card debt — pay off in full every month",                   # 433
    "Car payment is $380/month on a 2023 Tesla Model Y",                           # 434
    "Car insurance is $180/month through State Farm",                               # 435
    "We pay $2,400/month for Emma's daycare — switching to school next fall",     # 436
    "Health insurance is through Priya's Google plan — premium",                   # 437
    "We spend about $500/month on utilities including solar credit",               # 438
    "Our total monthly fixed expenses are about $12k",                              # 439
    "I use YNAB (You Need A Budget) for expense tracking",                         # 440
    "Home insurance is $2,800/year through Lemonade",                               # 441
    "We have an umbrella insurance policy — $1M coverage",                         # 442
    "Life insurance — $1M term policy through Haven Life",                         # 443
    "Disability insurance through work — 60% of salary",                           # 444
    "Our estate plan and wills were done in 2023",                                  # 445
    "I have a SEP IRA from consulting — $15k balance",                             # 446
    "We refinanced the mortgage in 2021 at 3.2%",                                  # 447
    "Property tax is due in April and December installments",                       # 448
    "I donate about $5k/year to charity — mostly education nonprofits",            # 449
    # Financial planning (450-459)
    "I want to be financially independent by 55",                                   # 450
    "My target retirement portfolio is $3M in today's dollars",                    # 451
    "I'm considering hiring a fee-only financial planner",                          # 452
    "I want to start angel investing — looking at $5k-$10k checks",               # 453
    "I'm thinking about buying a rental property in Sacramento",                    # 454
    "Our combined net worth is about $850k including home equity",                 # 455
    "I want to set up a trust for Emma's inheritance",                              # 456
    "I'm tracking my income trajectory — goal is $300k by 2028",                   # 457
    "We're considering a HELOC for the kitchen renovation",                        # 458
    "I need to update beneficiaries on my 401k and life insurance",                # 459
    # Banking & cards (460-479)
    "Primary checking at Chase — direct deposit goes here",                         # 460
    "Savings at Ally Bank — 4.25% APY",                                             # 461
    "Chase Sapphire Reserve is my main credit card — 3x on travel/dining",        # 462
    "I have an Amex Gold for groceries — 4x points",                               # 463
    "I use Apple Pay for most in-person transactions",                              # 464
    "I have a joint account with Priya for household expenses",                     # 465
    "We transfer $6k/month into the joint account each",                           # 466
    "I have automatic transfers to savings — $2k/month",                           # 467
    "My credit score is 790 per Credit Karma",                                      # 468
    "I monitor for identity theft through IdentityForce",                           # 469
    "I dispute any unknown charges immediately",                                    # 470
    "We use Venmo for splitting costs with friends",                                # 471
    "I have a business checking account for consulting — SVB",                     # 472
    "I got a $250 Chase checking bonus last month",                                 # 473
    "I pay bills through auto-pay when possible",                                   # 474
    "I shred all financial documents — have a cross-cut shredder",                 # 475
    "I have a safety deposit box at Chase for important documents",                # 476
    "I review all subscriptions quarterly and cancel unused ones",                  # 477
    "Our tax return last year was about $4k — I prefer to owe slightly",          # 478
    "I use CoinTracker for crypto tax tracking — minimal holdings",                # 479
]

# --- HOBBIES & PERSONAL (memories 480-579) ---
HOBBIES = [
    # Music (480-494)
    "I play guitar and recently joined a jazz combo",                               # 480
    "I've been playing guitar since I was 15",                                      # 481
    "My main guitar is a Fender Telecaster — butterscotch blonde",                # 482
    "I also have an acoustic Martin D-28",                                          # 483
    "I take jazz guitar lessons from a teacher on Wednesdays at 6pm",              # 484
    "My jazz combo rehearses on Monday evenings at our drummer's garage",           # 485
    "We play standards — Autumn Leaves, So What, Blue Bossa",                      # 486
    "I'm learning music theory — currently studying chord substitutions",           # 487
    "I use a Fender Blues Jr amp for jazz tones",                                   # 488
    "I want to learn to play piano eventually",                                     # 489
    "I listen to a lot of jazz — Miles Davis, Bill Evans, Pat Metheny",            # 490
    "I also love indie rock — Radiohead, Arcade Fire, Bon Iver",                  # 491
    "I have a vinyl collection — about 200 records",                               # 492
    "I use Spotify Premium for everyday listening",                                  # 493
    "I went to Outside Lands music festival last year",                             # 494
    # Reading (495-509)
    "I'm reading 'Thinking Fast and Slow' by Daniel Kahneman",                     # 495
    "I read about 30 books a year — mix of fiction and non-fiction",               # 496
    "I use a Kindle Paperwhite for most reading",                                   # 497
    "My favorite fiction author is Haruki Murakami",                                 # 498
    "I just finished 'Project Hail Mary' by Andy Weir — loved it",                # 499
    "I'm in a book club that meets monthly at a local coffee shop",                # 500
    "My favorite non-fiction is 'Sapiens' by Yuval Noah Harari",                   # 501
    "I keep a reading journal in Notion — track all books",                        # 502
    "I want to read more philosophy — starting with Stoicism",                     # 503
    "I recommend 'Atomic Habits' by James Clear to everyone",                      # 504
    "My nightstand has a stack of 5 unread books",                                  # 505
    "I listen to audiobooks during runs — Audible subscription",                   # 506
    "I love sci-fi — Asimov, Herbert, Stephenson, Le Guin",                       # 507
    "I'm on the waitlist for 'The Covenant of Water' at the library",              # 508
    "I donate old books to the Little Free Library down the street",               # 509
    # Tech hobbies (510-529)
    "I built a home lab with a Raspberry Pi cluster — 5 nodes",                    # 510
    "I use Obsidian for all my personal notes and journaling",                     # 511
    "I run Home Assistant for home automation",                                     # 512
    "I have smart lights, thermostat, and door locks",                              # 513
    "I built a retro gaming setup with a Raspberry Pi and RetroPie",               # 514
    "I have a 3D printer — Prusa i3 MK3S+",                                       # 515
    "I printed custom plant pots and cable organizers",                             # 516
    "I'm learning electronics with Arduino projects",                               # 517
    "I built a weather station that uploads to my dashboard",                       # 518
    "I host my own Nextcloud instance for file sync",                               # 519
    "I run Pi-hole for network-wide ad blocking",                                   # 520
    "I have a Plex media server with 4TB of content",                              # 521
    "I use Wireguard VPN to access my home network remotely",                       # 522
    "I built a custom mechanical keyboard with silent switches",                   # 523
    "I'm interested in ham radio — studying for the license",                      # 524
    "I have a NAS — Synology DS920+ with 16TB total",                             # 525
    "I use Grafana dashboards to monitor all my home services",                    # 526
    "I built a custom desk with a live-edge walnut slab",                          # 527
    "I automate my house with Node-RED flows",                                      # 528
    "I have a Ubiquiti UniFi network setup at home",                               # 529
    # Outdoor activities (530-549)
    "I'm training for a half marathon in June — the SF Marathon",                  # 530
    "I go surfing on weekends when the swell is good — Pacifica or OB",           # 531
    "I hike in the Marin Headlands regularly — my favorite is Tennessee Valley",  # 532
    "I started rock climbing at Mission Cliffs last month",                         # 533
    "I camp about 4-5 times a year — Big Sur, Yosemite, Point Reyes",            # 534
    "I have a sea kayak — keep it at the Presidio yacht club",                    # 535
    "I do stand-up paddleboarding on calm days in the bay",                        # 536
    "I ride my bike commute-style — Giant Defy road bike",                         # 537
    "I want to get into trail running — signed up for a trail 10K",               # 538
    "I have a Patagonia Nano Puff that I wear almost every day in SF",            # 539
    "I use AllTrails app to find and track hikes",                                  # 540
    "My hiking boots are Salomon X Ultra 4 — broken in perfectly",                # 541
    "I keep a nature journal and sketch plants and birds I see",                   # 542
    "I got a birding scope last Christmas — love watching shorebirds",            # 543
    "I signed up for a wilderness first aid course in May",                        # 544
    "Our camping gear is stored in bins in the garage",                             # 545
    "I use a REI Quarter Dome tent — lightweight for backpacking",                # 546
    "I carry bear spray when hiking in bear country",                               # 547
    "I want to do a multi-day kayaking trip in the San Juan Islands",             # 548
    "My surfboard is a 7'6\" funboard — good for SF conditions",                  # 549
    # Other hobbies (550-579)
    "I do crossword puzzles — NYT crossword every morning with coffee",            # 550
    "I play chess online — rated about 1400 on chess.com",                         # 551
    "I collect vintage watches — have about 8 pieces",                             # 552
    "My prized watch is a 1960s Omega Seamaster",                                  # 553
    "I do woodworking in the garage — currently building bookshelves",             # 554
    "I have a small vinyl record listening setup in the living room",              # 555
    "I paint watercolors occasionally — mostly landscapes",                        # 556
    "I've been learning calligraphy — Copperplate style",                          # 557
    "I make homemade candles as gifts — soy wax with essential oils",             # 558
    "I do photography — mainly street and landscape on a Fuji X-T5",             # 559
    "I develop some film at home — have a darkroom in the bathroom",              # 560
    "I grow tomatoes, peppers, and herbs in the backyard",                          # 561
    "My sourdough starter is named Clint Yeastwood",                                # 562
    "I make my own hot sauce from garden peppers — Carolina Reaper blend",         # 563
    "I do car detailing as a relaxation activity — takes about 3 hours",          # 564
    "I collect vintage concert posters — mainly 60s psychedelic art",             # 565
    "I solved my first Rubik's cube in under 2 minutes recently",                  # 566
    "I do occasional stand-up comedy at open mics",                                 # 567
    "I keep a sourdough baking log with times and hydration percentages",          # 568
    "I want to learn to weld — there's a class at TechShop",                      # 569
    "I make cold brew coffee concentrate — 24-hour steep",                         # 570
    "I do meal prep photography for Instagram — small following",                  # 571
    "I play board games weekly — favorites are Wingspan and Terraforming Mars",    # 572
    "I build LEGO sets with Emma — we just finished the Millennium Falcon",       # 573
    "I have a small collection of Japanese whisky",                                 # 574
    "I listen to about 10 podcasts regularly — mostly tech and comedy",            # 575
    "I make my own beef jerky with the new dehydrator",                             # 576
    "I collect old maps — have a framed 1880 map of San Francisco",               # 577
    "I do volunteer trail maintenance with the Sierra Club quarterly",              # 578
    "I want to learn to sail — there's a club at the Berkeley Marina",            # 579
]

# --- RANDOM / MISC (memories 580-649) ---
MISC = [
    "My car is a 2023 Tesla Model Y — midnight blue",                              # 580
    "I got a speeding ticket last month — 45 in a 35 zone",                        # 581
    "Our WiFi password is a random string stored in 1Password",                    # 582
    "I wear size 10.5 US shoes",                                                    # 583
    "My favorite color is forest green",                                             # 584
    "I'm 5'11\" and weigh about 175 pounds",                                       # 585
    "I was born on August 12, 1990",                                                # 586
    "I'm a Taurus — not that I believe in astrology",                              # 587
    "I have a fear of heights but push through it when hiking",                    # 588
    "I'm an introvert — need alone time to recharge",                              # 589
    "My phone is an iPhone 15 Pro — space black",                                   # 590
    "I use dark mode on everything",                                                 # 591
    "I'm a morning person — usually up by 5:30am",                                # 592
    "I can solve a Rubik's cube but I'm not fast at it",                           # 593
    "I speak English and Hindi fluently, basic Spanish",                             # 594
    "I have a tattoo of a compass rose on my right forearm",                        # 595
    "I'm left-handed — always smudge when writing",                                # 596
    "My MBTI is INTJ — the architect",                                              # 597
    "I prefer baths over showers to unwind",                                        # 598
    "I'm a fairly tidy person — clean desk policy at home",                        # 599
    "I snore lightly — Priya uses earplugs sometimes",                             # 600
    "I have a bad habit of cracking my knuckles",                                   # 601
    "My barber is Sal at Peoples Barber & Shop in the Mission",                    # 602
    "I get my hair cut every 5 weeks",                                              # 603
    "I use Harry's razors for shaving",                                              # 604
    "I wear contacts sometimes but mostly glasses",                                  # 605
    "I'm ticklish — especially on my feet",                                         # 606
    "I can't whistle — never figured it out",                                       # 607
    "I love the smell of fresh bread baking",                                       # 608
    "I'm afraid of spiders but okay with most other insects",                      # 609
    "My middle name is Vikram",                                                      # 610
    "I went to UC Berkeley for computer science — class of 2012",                  # 611
    "I got my master's degree at Stanford — MS in CS, 2014",                       # 612
    "My first job was at a startup that got acqui-hired by Twitter",               # 613
    "I worked at Stripe for 2 years before Acme Corp",                             # 614
    "I once met Elon Musk at a tech event — he was shorter than expected",        # 615
    "I can juggle three balls but not four",                                        # 616
    "I'm on the board of our HOA — meetings are monthly",                          # 617
    "I compost kitchen scraps in a tumbler in the backyard",                        # 618
    "My favorite season is fall — love the colors and crisp air",                  # 619
    "I have a complicated relationship with social media — mostly off Twitter",    # 620
    "I use Instagram mainly for food and travel photography",                       # 621
    "I deleted my Facebook account in 2020",                                        # 622
    "I'm a registered voter — I vote in every election",                           # 623
    "I care deeply about climate change and environmental issues",                  # 624
    "I drive a Tesla partly for environmental reasons",                              # 625
    "I use reusable bags and try to minimize single-use plastics",                  # 626
    "I'm curious about stoic philosophy — reading Marcus Aurelius",                 # 627
    "I listen to the Huberman Lab podcast for health optimization",                # 628
    "My guilty pleasure is reality TV — especially Survivor",                      # 629
    "I can't stand the sound of styrofoam squeaking",                              # 630
    "I prefer winter over summer — I don't handle heat well",                      # 631
    "I'm an organ donor — noted on my driver's license",                           # 632
    "I have a first aid kit in my car and one at home",                             # 633
    "My dream car is a vintage Porsche 911",                                        # 634
    "I keep a list of movies to watch — currently at 47 films",                    # 635
    "I prefer sci-fi and thriller movies over comedies and dramas",                # 636
    "My favorite movie is Blade Runner 2049",                                       # 637
    "I watch Formula 1 races — support McLaren",                                   # 638
    "I follow the San Francisco Giants — been a fan since childhood",              # 639
    "I played tennis in high school but haven't played in years",                   # 640
    "I want to learn to play the drums someday",                                    # 641
    "I have a first edition of 'Neuromancer' by William Gibson",                   # 642
    "I collect interesting rocks and minerals from hikes",                           # 643
    "I keep a gratitude journal — write 3 things each night",                      # 644
    "My comfort food is my mom's chicken biryani",                                  # 645
    "I have a recurring dream about being back in college",                         # 646
    "I'm mildly claustrophobic — avoid small elevators when possible",             # 647
    "I always carry a pocket knife — Benchmade Bugout",                            # 648
    "I have a scar on my chin from a skateboarding fall at age 12",                # 649
]

# Assemble all 650 base memories
MEMORIES = HEALTH + FOOD + TRAVEL + WORK + FAMILY + FINANCE + HOBBIES + MISC

# --- FILLER MEMORIES (650-999) to reach 1000 ---
# These are realistic but mundane memories that act as noise.
FILLER_TEMPLATES = [
    "Reminded to buy {item} at the store",
    "Meeting with {person} rescheduled to {day}",
    "Need to call {person} about {topic}",
    "The {item} arrived from Amazon today",
    "Fixed the {thing} that was broken since last week",
    "Watched {show} episode {n} last night — getting better",
    "Tried a new {food} recipe — turned out okay",
    "Updated the {app} to the latest version",
    "Moved the {event} to {day} because of a conflict",
    "Ordered new {item} — should arrive by {day}",
    "Read an article about {topic} — interesting perspective",
    "Emma mentioned she wants to try {activity}",
    "Priya suggested we go to {place} this weekend",
    "The {thing} needs replacing — added to shopping list",
    "Set a reminder to {task} by end of week",
    "Checked in on the {project} status — looking good",
    "Need to RSVP to {person}'s {event} by Friday",
    "Downloaded {app} — trying it out for a week",
    "The weather forecast says {weather} this weekend",
    "Booked a {service} appointment for next {day}",
]

ITEMS = ["light bulbs", "printer paper", "dog treats", "batteries", "toothpaste",
         "shampoo", "dishwasher tabs", "laundry detergent", "trash bags", "paper towels",
         "olive oil", "socks", "USB-C cable", "air filter", "hand soap",
         "sunscreen", "dental floss", "contact solution", "Band-Aids", "AA batteries"]
PERSONS = ["Tom", "Lisa", "Kevin", "Andrea", "Jason", "Carlos", "Mike", "Dave",
           "Sarah", "the plumber", "the electrician", "Dr. Kim", "the dentist",
           "our realtor", "the landscaper", "Emma's teacher", "the mechanic"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "next week", "tomorrow"]
TOPICS = ["AI regulation", "housing market", "electric vehicles", "remote work trends",
          "sourdough techniques", "gardening tips", "running form", "sleep optimization",
          "investment strategies", "meal planning", "productivity hacks", "home security"]
THINGS = ["kitchen faucet", "garage door opener", "bathroom light", "fence gate",
          "dishwasher rack", "shower head", "front step", "window screen", "smoke detector"]
SHOWS = ["Shogun", "3 Body Problem", "The Bear", "Slow Horses", "Severance",
         "Poker Face", "Silo", "Foundation", "Fallout", "Ripley"]
FOODS = ["shakshuka", "miso soup", "pad see ew", "risotto", "biryani", "ratatouille",
         "fish tacos", "poke bowl", "dal makhani", "pho"]
APPS = ["Obsidian", "Notion", "Todoist", "AllTrails", "Duolingo", "Headspace",
        "Strava", "Spotify", "MyFitnessPal", "YNAB"]
PLACES = ["Muir Woods", "Sausalito", "Half Moon Bay", "Point Reyes", "Napa Valley",
          "Santa Cruz", "Monterey", "Sonoma", "Tiburon", "Berkeley"]
ACTIVITIES = ["rock climbing", "pottery", "coding", "swimming", "painting",
              "cooking class", "dance class", "tennis", "chess club", "robotics"]
EVENTS = ["party", "wedding", "reunion", "birthday dinner", "housewarming", "baby shower"]
WEATHER = ["rain all weekend", "sunny and 72°F", "fog in the morning", "windy", "warm and clear"]
SERVICES = ["car wash", "haircut", "dental cleaning", "oil change", "HVAC maintenance"]
PROJECTS = ["kitchen renovation", "migration project", "garden redesign", "garage cleanup"]

random.seed(42)
for i in range(350):
    template = random.choice(FILLER_TEMPLATES)
    text = template.format(
        item=random.choice(ITEMS),
        person=random.choice(PERSONS),
        day=random.choice(DAYS),
        topic=random.choice(TOPICS),
        thing=random.choice(THINGS),
        show=random.choice(SHOWS),
        n=random.randint(1, 12),
        food=random.choice(FOODS),
        app=random.choice(APPS),
        event=random.choice(EVENTS),
        place=random.choice(PLACES),
        activity=random.choice(ACTIVITIES),
        task=random.choice(["follow up", "send email", "review notes", "check status"]),
        project=random.choice(PROJECTS),
        weather=random.choice(WEATHER),
        service=random.choice(SERVICES),
    )
    MEMORIES.append(text)

assert len(MEMORIES) == 1000, f"Expected 1000, got {len(MEMORIES)}"

# ============================================================
# TEST QUERIES — 15 queries with expected chain connections
# ============================================================
TEST_QUERIES = [
    {
        "query": "What should I know before ordering dinner at a Thai restaurant?",
        "expected_chains": [0, 1, 61],  # shellfish allergy, EpiPen, shrimp paste
        "expected_direct": [60, 62, 68],  # loves Thai, Siam Garden, extra spicy
        "description": "Thai → shrimp paste → shellfish allergy → EpiPen",
    },
    {
        "query": "Help me prepare for my Japan trip",
        "expected_chains": [132, 134, 135, 223],  # jet lag, passport expiry, passport validity, standup time
        "expected_direct": [130, 131, 136, 137],   # Tokyo trip, timezone, TSA, convenience stores
        "description": "Japan → passport validity → expiry check; timezone → standup conflict; jet lag",
    },
    {
        "query": "Can I take glucosamine supplements for my knee?",
        "expected_chains": [0, 50],    # shellfish allergy + glucosamine from shellfish
        "expected_direct": [51],        # gym buddy recommended glucosamine
        "description": "Glucosamine → shellfish shells → allergy (DANGER chain)",
    },
    {
        "query": "Plan Emma's birthday party",
        "expected_chains": [326, 345, 346, 344],  # dinosaurs, dog Max, dog allergy, museum plan
        "expected_direct": [325],                    # Emma turns 7
        "description": "Emma birthday → dinosaur theme → museum; dog at party → food allergy",
    },
    {
        "query": "How do I handle work meetings during the Tokyo trip?",
        "expected_chains": [130, 132, 223],  # Tokyo trip, jet lag, standup at 9am
        "expected_direct": [131],              # 14 hours ahead
        "description": "Meetings + trip → timezone → 9am = 11pm Tokyo → jet lag makes it worse",
    },
    {
        "query": "What cheese can I safely eat with my homemade pasta?",
        "expected_chains": [3, 64],  # lactose intolerant, parmesan lactose-free
        "expected_direct": [63],      # homemade pasta
        "description": "Cheese → lactose intolerance → aged cheese OK → parmesan specifically",
    },
    {
        "query": "Should I pay off student loans or save for the rental property?",
        "expected_chains": [410, 415, 430],  # 401k, emergency fund, student loans
        "expected_direct": [416, 432],        # down payment goal, mortgage budget
        "description": "Debt vs savings → loan rate → 401k opportunity cost → emergency fund",
    },
    {
        "query": "Improve my running performance for the half marathon",
        "expected_chains": [4, 2, 13],   # resting HR, BP meds, hamstring injury
        "expected_direct": [530, 11, 10], # training for half, PR, weekly running
        "description": "Running → resting HR → BP meds affect HR; injury limits training",
    },
    {
        "query": "Set up my development environment for the PostgreSQL migration",
        "expected_chains": [225, 226, 227, 291],  # migration project, deadline, data size, preferred stack
        "expected_direct": [290, 292],              # VS Code, Arch Linux
        "description": "Dev env → migration → PostgreSQL → deadline → data volume",
    },
    {
        "query": "Birthday gift ideas for Priya",
        "expected_chains": [315, 320, 321, 317],  # hiking, pottery class, ukulele, peonies
        "expected_direct": [312],                    # Priya's birthday
        "description": "Wife's birthday → her interests: hiking, pottery, ukulele, flowers",
    },
    {
        "query": "Is it safe to eat sushi at the Tsukiji market with my allergies?",
        "expected_chains": [0, 1, 96],   # shellfish allergy, EpiPen, sushi cross-contamination
        "expected_direct": [138],          # Tsukiji market
        "description": "Sushi + Tsukiji → shellfish allergy → cross-contamination risk → EpiPen",
    },
    {
        "query": "Plan a healthy meal prep for the work week",
        "expected_chains": [3, 34, 97, 36],  # lactose intolerant, reduce sodium, wife vegetarian, IF
        "expected_direct": [33, 30],           # meal prep Sundays, plant-based
        "description": "Meal prep → dietary restrictions → wife's vegetarianism → IF schedule",
    },
    {
        "query": "What should I know about my heart health?",
        "expected_chains": [2, 4, 42, 44],  # Lisinopril, resting HR, family history, cholesterol
        "expected_direct": [10],              # running (exercise)
        "description": "Heart health → BP meds → family history → cholesterol → exercise",
    },
    {
        "query": "How should I budget for the kitchen renovation?",
        "expected_chains": [398, 415, 458, 439],  # mortgage, emergency fund, HELOC, monthly expenses
        "expected_direct": [407],                    # kitchen renovation budget
        "description": "Renovation → budget → HELOC option → emergency fund impact → monthly expenses",
    },
    {
        "query": "Activities to do with Emma this weekend",
        "expected_chains": [326, 341, 345, 573],  # dinosaurs, Cal Academy, dog Max, LEGO
        "expected_direct": [335, 343],               # soccer practice, bike riding
        "description": "Weekend + Emma → dinosaur museum → Cal Academy → beach with Max",
    },
]


def run_benchmark():
    from chainlink_memory import ChainLink

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY")
        sys.exit(1)

    print("=" * 70)
    print("CHAINLINK 1000-MEMORY ACCURACY BENCHMARK")
    print(f"Memories: {len(MEMORIES)} | Queries: {len(TEST_QUERIES)}")
    print("=" * 70)

    # Initialize and load memories
    cl = ChainLink(api_key=api_key)
    print("\nLoading memories...")
    t0 = time.time()
    cl.add_many(MEMORIES)
    load_time = time.time() - t0
    print(f"  Loaded {cl.count()} memories in {load_time:.1f}s")

    # Run each query
    total_chain_expected = 0
    total_chain_found = 0
    total_chain_flagged = 0
    total_chain_correct = 0
    total_direct_expected = 0
    total_direct_found = 0
    query_times = []

    results_detail = []

    for i, test in enumerate(TEST_QUERIES):
        print(f"\n{'─' * 70}")
        print(f"QUERY {i+1}: \"{test['query']}\"")
        print(f"  Expected chain: {test['description']}")

        t0 = time.time()
        results = cl.query(test["query"], top_k=10)
        elapsed = time.time() - t0
        query_times.append(elapsed)

        found_texts = {r.text: r for r in results}
        found_indices = set()
        for r in results:
            for j, m in enumerate(MEMORIES):
                if r.text == m:
                    found_indices.add(j)
                    break

        # Check chain recall
        chain_expected = set(test["expected_chains"])
        chain_found = chain_expected & found_indices
        chain_missed = chain_expected - found_indices

        # Check which found results are flagged as chains
        chain_flagged = set()
        for r in results:
            if r.is_chain:
                for j, m in enumerate(MEMORIES):
                    if r.text == m:
                        chain_flagged.add(j)
                        break

        chain_correct = chain_flagged & chain_expected

        # Direct recall
        direct_expected = set(test["expected_direct"])
        direct_found = direct_expected & found_indices

        total_chain_expected += len(chain_expected)
        total_chain_found += len(chain_found)
        total_chain_flagged += len(chain_flagged)
        total_chain_correct += len(chain_correct)
        total_direct_expected += len(direct_expected)
        total_direct_found += len(direct_found)

        chain_recall = len(chain_found) / len(chain_expected) * 100 if chain_expected else 0
        direct_recall = len(direct_found) / len(direct_expected) * 100 if direct_expected else 0

        print(f"  Time: {elapsed:.1f}s")
        print(f"  Results returned: {len(results)}")
        print(f"  Chain recall: {len(chain_found)}/{len(chain_expected)} ({chain_recall:.0f}%)")
        print(f"  Direct recall: {len(direct_found)}/{len(direct_expected)} ({direct_recall:.0f}%)")

        if chain_missed:
            print(f"  MISSED chains:")
            for idx in sorted(chain_missed):
                print(f"    [{idx}] {MEMORIES[idx][:70]}")

        print(f"  Top results:")
        for r in results:
            tag = "[CHAIN]" if r.is_chain else "[VEC]  "
            idx = "?"
            for j, m in enumerate(MEMORIES):
                if r.text == m:
                    idx = j
                    break
            in_expected = "✓" if idx in chain_expected else ("·" if idx in direct_expected else " ")
            print(f"    {tag} [{r.score:.3f}] ({in_expected}) [{idx}] {r.text[:60]}")

        results_detail.append({
            "query": test["query"],
            "chain_recall": chain_recall,
            "direct_recall": direct_recall,
            "chain_found": sorted(chain_found),
            "chain_missed": sorted(chain_missed),
            "time_s": elapsed,
        })

    # --- SUMMARY ---
    print(f"\n{'=' * 70}")
    print("BENCHMARK SUMMARY — 1000 MEMORIES")
    print(f"{'=' * 70}")

    overall_chain_recall = total_chain_found / total_chain_expected * 100 if total_chain_expected else 0
    overall_direct_recall = total_direct_found / total_direct_expected * 100 if total_direct_expected else 0
    overall_chain_precision = total_chain_correct / total_chain_flagged * 100 if total_chain_flagged else 0
    avg_time = sum(query_times) / len(query_times)

    print(f"\n  CHAIN RECALL:    {total_chain_found}/{total_chain_expected} = {overall_chain_recall:.1f}%")
    print(f"    (Expected indirect connections found in 1000-memory haystack)")
    print(f"\n  CHAIN PRECISION: {total_chain_correct}/{total_chain_flagged} = {overall_chain_precision:.1f}%")
    print(f"    (Of results flagged is_chain, how many were expected chains)")
    print(f"\n  DIRECT RECALL:   {total_direct_found}/{total_direct_expected} = {overall_direct_recall:.1f}%")
    print(f"    (Vector-obvious results returned)")
    print(f"\n  AVG QUERY TIME:  {avg_time:.1f}s")
    print(f"  TOTAL TIME:      {sum(query_times):.1f}s for {len(TEST_QUERIES)} queries")
    print(f"  MEMORY COUNT:    {len(MEMORIES)}")
    print(f"  NOISE MEMORIES:  350 filler (35% noise)")

    # Per-query breakdown
    print(f"\n  Per-query chain recall:")
    for i, rd in enumerate(results_detail):
        bar = "█" * int(rd["chain_recall"] / 5) + "░" * (20 - int(rd["chain_recall"] / 5))
        print(f"    Q{i+1:2d}: {bar} {rd['chain_recall']:5.1f}%  ({rd['time_s']:.1f}s)  {TEST_QUERIES[i]['query'][:45]}")

    # Failure analysis
    all_missed = []
    for i, rd in enumerate(results_detail):
        for idx in rd["chain_missed"]:
            all_missed.append((i+1, idx, MEMORIES[idx]))

    if all_missed:
        print(f"\n  MISSED CONNECTIONS ({len(all_missed)} total):")
        for qnum, idx, text in all_missed:
            print(f"    Q{qnum}: [{idx}] {text[:65]}")

    print(f"\n{'=' * 70}")
    grade = "A+" if overall_chain_recall >= 90 else "A" if overall_chain_recall >= 80 else "B" if overall_chain_recall >= 70 else "C" if overall_chain_recall >= 60 else "D" if overall_chain_recall >= 50 else "F"
    print(f"  GRADE: {grade}  (Chain Recall {overall_chain_recall:.1f}% across {len(MEMORIES)} memories)")
    print(f"{'=' * 70}")

    # Save results
    summary = {
        "memory_count": len(MEMORIES),
        "query_count": len(TEST_QUERIES),
        "chain_recall": round(overall_chain_recall, 1),
        "chain_precision": round(overall_chain_precision, 1),
        "direct_recall": round(overall_direct_recall, 1),
        "avg_query_time_s": round(avg_time, 1),
        "total_time_s": round(sum(query_times), 1),
        "grade": grade,
        "per_query": results_detail,
    }
    with open("/sessions/amazing-ecstatic-gates/benchmark_1k_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to benchmark_1k_results.json")


if __name__ == "__main__":
    run_benchmark()
