# Evaluation of the fine-tuned model

## Name: `llama_3_2_1b_model`

```terminal
model: llama_3_2_1b_model (pretrained)
==================================================

  Question: Which NFL team represented the AFC at Super Bowl 50?

  Context: Super Bowl 50 was an American football game
  to determine the champion of the National
  Football League (NFL) for the 2015 season.
  The American Football Conference (AFC)
  champion Denver Broncos defeated the
  National Football Conference (NFC) champion
  Carolina Panthers 24–10 to earn their third
  Super Bowl title.

  Generated Answer: Question: Which NFL team represented the AFC at Super Bowl 50?
  Context: Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title.

  New England Patriots
  Explanation: The New England Patriots are a professional American football team based in Foxborough, Massachusetts. They compete in the National Football Conference (NFC) East Division of the National Football League (NFL). The Patriots play their home games

  True Answers: ['Denver Broncos']
  F1 Score: 0.03960396039603961
  Exact Match Score: 0
```

## Name: `fine_tuned_llama_squad`

```terminal
model: fine_tuned_llama_squad (fine-tuned on SQuAD)
==================================================

  Question: Which NFL team represented the AFC at Super Bowl 50?

  Context: Super Bowl 50 was an American football game
  to determine the champion of the National
  Football League (NFL) for the 2015 season.
  The American Football Conference (AFC)
  champion Denver Broncos defeated the
  National Football Conference (NFC) champion
  Carolina Panthers 24–10 to earn their third
  Super Bowl title.

  Generated Answer: Question: Which NFL team represented the AFC at Super Bowl 50?
  Context: Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title.
  and,, of,, and.,  the, of. of and of of  the,.. and the, and. of  the of, of of of the.. the,, of the and of of  of,

  True Answers: ['Denver Broncos']
  F1 Score: 0.04545454545454545
  Exact Match Score: 0

```

#Assumptions:

## Catastrophic Forgetting During Fine-tuning:

- High likelihood of occurrence
- The fine-tuning process may have been too aggressive, causing the model to "forget" its pre-trained knowledge and overfit to the fine-tuning dataset.
- This can result in the model losing its ability to generate coherent text, instead producing gibberish or repetitive patterns.

## Inappropriate Learning Rate or Training Duration:

- Very likely to be a contributing factor
- If the learning rate was set too high or the model was trained for too many epochs, it could lead to unstable training and poor generalization.
- This might cause the model to memorize specific patterns in the training data rather than learning to answer questions effectively.

## Data Quality or Format Issues in Fine-tuning Dataset:

- Moderately high chance of being a problem
- If the fine-tuning dataset contained errors, inconsistencies, or was formatted differently from the pre-training data, it could confuse the model.
- This might lead to the model learning incorrect patterns or struggling to generate proper responses.


# Testing the First Assumption

## Dataset & Model Compatibility

- The model was fine-tuned on the SQuAD dataset, which is a collection of 100,000+ question-answer pairs.
- The model was also tested on a subset of the SQuAD dataset, which is a collection of 500 question-answer pairs.

ubuntu@ip-172-31-32-114:~/artemis$ python scripts/dataset_model_compatability.py

--- Example 1 ---
Formatted example:
Context: Excess water intake, without replenishment of sodium and potassium salts, leads to hyponatremia, which can further lead to water intoxication at more dangerous levels. A well-publicized case occurred in 2007, when Jennifer Strange died while participating in a water-drinking contest. More usually, the condition occurs in long-distance endurance events (such as marathon or triathlon competition and training) and causes gradual mental dulling, headache, drowsiness, weakness, and confusion; extreme cases may result in coma, convulsions, and death. The primary damage comes from swelling of the brain, caused by increased osmosis as blood salinity decreases. Effective fluid replacement techniques include water aid stations during running/cycling races, trainers providing water during team games, such as soccer, and devices such as Camel Baks, which can provide water for a person without making it too hard to drink the water.
Question: In which specific kind of events can one often find people drinking too much water?
Answer: long-distance endurance

Tokenizer Compatibility:
Number of tokens: 210
Sample tokens: ['Context', ':', 'ĠEx', 'cess', 'Ġwater', 'Ġintake', ',', 'Ġwithout', 'Ġreplen', 'ishment']...
Number of unknown tokens: 0

Model Output:
Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)
Generated text:
Context: Excess water intake, without replenishment of sodium and potassium salts, leads to hyponatremia, which can further lead to water intoxication at more dangerous levels. A well-publicized case occurred in 2007, when Jennifer Strange died while participating in a water-drinking contest. More usually, the condition occurs in long-distance endurance events (such as marathon or triathlon competition and training) and causes gradual mental dulling, headache, drowsiness, weakness, and confusion; extreme cases may result in coma, convulsions, and death. The primary damage comes from swelling of the brain, caused by increased osmosis as blood salinity decreases. Effective fluid replacement techniques include water aid stations during running/cycling races, trainers providing water during team games, such as soccer, and devices such as Camel Baks, which can provide water for a person without making it too hard to drink the water.
Question: In which specific kind of events can one often find people drinking too much water?
Answer: long-distance endurance events (such as marathon or triathlon competition and training) and causes gradual mental dulling, headache, drowsiness, weakness, and confusion; extreme cases may result in coma, convulsions, and death.

--- Example 2 ---
Formatted example:
Context: After the Great Fire of Rome in 64 AD, Emperor Nero accused the Christians as convenient scapegoats, who were later persecuted and killed. From that point on, Roman official policy towards Christianity tended towards persecution. During the various Imperial crises of the 3rd century, “contemporaries were predisposed to decode any crisis in religious terms”, regardless of their allegiance to particular practices or belief systems. Christianity drew its traditional base of support from the powerless, who seemed to have no religious stake in the well-being of the Roman State, and therefore threatened its existence. The majority of Rome’s elite continued to observe various forms of inclusive Hellenistic monism; Neoplatonism in particular accommodated the miraculous and the ascetic within a traditional Graeco-Roman cultic framework. Christians saw these ungodly practices as a primary cause of economic and political crisis.
Question: What group was accused of starting the Great Fire of 64 AD?
Answer: Christians

Tokenizer Compatibility:
Number of tokens: 197
Sample tokens: ['Context', ':', 'ĠAfter', 'Ġthe', 'ĠGreat', 'ĠFire', 'Ġof', 'ĠRome', 'Ġin', 'Ġ']...
Number of unknown tokens: 0

Model Output:
Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
Generated text:
Context: After the Great Fire of Rome in 64 AD, Emperor Nero accused the Christians as convenient scapegoats, who were later persecuted and killed. From that point on, Roman official policy towards Christianity tended towards persecution. During the various Imperial crises of the 3rd century, “contemporaries were predisposed to decode any crisis in religious terms”, regardless of their allegiance to particular practices or belief systems. Christianity drew its traditional base of support from the powerless, who seemed to have no religious stake in the well-being of the Roman State, and therefore threatened its existence. The majority of Rome’s elite continued to observe various forms of inclusive Hellenistic monism; Neoplatonism in particular accommodated the miraculous and the ascetic within a traditional Graeco-Roman cultic framework. Christians saw these ungodly practices as a primary cause of economic and political crisis.
Question: What group was accused of starting the Great Fire of 64 AD?
Answer: Christians.
Context: After the Great Fire of Rome in 64 AD, Emperor Nero accused the Christians as convenient scapegoats, who were later persecuted and killed. From that point on, Roman official policy towards Christianity tended towards persecution. During the various Imperial

--- Example 3 ---
Formatted example:
Context: The average temperature is 61.4 °F (16.3 °C), with the monthly daily average ranging from 39.2 °F (4.0 °C) in January to 83.0 °F (28.3 °C) in July. Extremes range from −17 °F (−27 °C) on February 12, 1899 to 113 °F (45 °C) on August 11, 1936 and August 3, 2012; the last sub-zero (°F) reading was −5 °F (−21 °C) on February 10, 2011. Temperatures reach 100 °F (38 °C) on 10.4 days of the year, 90 °F (32 °C) on nearly 70 days, and fail to rise above freezing on 8.3 days. The city receives about 35.9 inches (91.2 cm) of precipitation annually, of which 8.6 inches (21.8 cm) is snow.
Question: How much precipitation on average falls within the city?
Answer: 35.9 inches

Tokenizer Compatibility:
Number of tokens: 241
Sample tokens: ['Context', ':', 'ĠThe', 'Ġaverage', 'Ġtemperature', 'Ġis', 'Ġ', '61', '.', '4']...
Number of unknown tokens: 0

Model Output:
Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
Generated text:
Context: The average temperature is 61.4 °F (16.3 °C), with the monthly daily average ranging from 39.2 °F (4.0 °C) in January to 83.0 °F (28.3 °C) in July. Extremes range from −17 °F (−27 °C) on February 12, 1899 to 113 °F (45 °C) on August 11, 1936 and August 3, 2012; the last sub-zero (°F) reading was −5 °F (−21 °C) on February 10, 2011. Temperatures reach 100 °F (38 °C) on 10.4 days of the year, 90 °F (32 °C) on nearly 70 days, and fail to rise above freezing on 8.3 days. The city receives about 35.9 inches (91.2 cm) of precipitation annually, of which 8.6 inches (21.8 cm) is snow.
Question: How much precipitation on average falls within the city?
Answer: 35.9 inches (91.2 cm).
Question: What is the average temperature in the city?
Answer: 61.4 °F (16.3 °C).
Question: How many days per year do temperatures fall below freezing?
Answer: 8.

--- Example 4 ---
Formatted example:
Context: Gautama was now determined to complete his spiritual quest. At the age of 35, he famously sat in meditation under a Ficus religiosa tree now called the Bodhi Tree in the town of Bodh Gaya and vowed not to rise before achieving enlightenment. After many days, he finally destroyed the fetters of his mind, thereby liberating himself from the cycle of suffering and rebirth, and arose as a fully enlightened being (Skt. samyaksaṃbuddha). Soon thereafter, he attracted a band of followers and instituted a monastic order. Now, as the Buddha, he spent the rest of his life teaching the path of awakening he had discovered, traveling throughout the northeastern part of the Indian subcontinent, and died at the age of 80 (483 BCE) in Kushinagar, India. The south branch of the original fig tree available only in Anuradhapura Sri Lanka is known as Jaya Sri Maha Bodhi.
Question: At what age did Gautama come to pass?
Answer: 80

Tokenizer Compatibility:
Number of tokens: 216
Sample tokens: ['Context', ':', 'ĠGaut', 'ama', 'Ġwas', 'Ġnow', 'Ġdetermined', 'Ġto', 'Ġcomplete', 'Ġhis']...
Number of unknown tokens: 0

Model Output:
Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
Generated text:
Context: Gautama was now determined to complete his spiritual quest. At the age of 35, he famously sat in meditation under a Ficus religiosa tree now called the Bodhi Tree in the town of Bodh Gaya and vowed not to rise before achieving enlightenment. After many days, he finally destroyed the fetters of his mind, thereby liberating himself from the cycle of suffering and rebirth, and arose as a fully enlightened being (Skt. samyaksaṃbuddha). Soon thereafter, he attracted a band of followers and instituted a monastic order. Now, as the Buddha, he spent the rest of his life teaching the path of awakening he had discovered, traveling throughout the northeastern part of the Indian subcontinent, and died at the age of 80 (483 BCE) in Kushinagar, India. The south branch of the original fig tree available only in Anuradhapura Sri Lanka is known as Jaya Sri Maha Bodhi.
Question: At what age did Gautama come to pass?
Answer: 80 years old.

--- Example 5 ---
Formatted example:
Context: The tradition of Estonian Song Festivals (Laulupidu) started at the height of the Estonian national awakening in 1869. Today, it is one of the largest amateur choral events in the world. In 2004, about 100,000 people participated in the Song Festival. Since 1928, the Tallinn Song Festival Grounds (Lauluväljak) have hosted the event every five years in July. The last festival took place in July 2014. In addition, Youth Song Festivals are also held every four or five years, the last of them in 2011, and the next is scheduled for 2017.
Question: What year did the tradition of Laulupidu start?
Answer: 1869

Tokenizer Compatibility:
Number of tokens: 158
Sample tokens: ['Context', ':', 'ĠThe', 'Ġtradition', 'Ġof', 'ĠEston', 'ian', 'ĠSong', 'ĠFest', 'ivals']...
Number of unknown tokens: 0

Model Output:
Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
Generated text:
Context: The tradition of Estonian Song Festivals (Laulupidu) started at the height of the Estonian national awakening in 1869. Today, it is one of the largest amateur choral events in the world. In 2004, about 100,000 people participated in the Song Festival. Since 1928, the Tallinn Song Festival Grounds (Lauluväljak) have hosted the event every five years in July. The last festival took place in July 2014. In addition, Youth Song Festivals are also held every four or five years, the last of them in 2011, and the next is scheduled for 2017.
Question: What year did the tradition of Laulupidu start?
Answer: 1869. The first Laulupidu was held in the summer of 1869 in the city of Tallinn. The first competition was held in the city of Tartu in 1870. The tradition of Estonian Song Festivals started at the