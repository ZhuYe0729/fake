# Llama3.1-8B-Instruct dense-BF16 prompt-template diagnosis

This is a diagnostic only. It does not replace legacy/common main-experiment results.
Both arms use identical fixed PMPD prefixes (100 examples/task), vLLM BF16, greedy decoding, max_new_tokens=256, max_input_tokens=3840, and explicit EOS+EOT stops. Only prompt construction differs.
The tokenizer has eos_token_id=128009 and eot_token_id=128009; both arms therefore use the same stop_token_ids=[128009].

| task | legacy metric | native metric | legacy avg new tokens | native avg new tokens | legacy marker continuations | native marker continuations |
|---|---:|---:|---:|---:|---:|---:|
| cnn_dm_1000 | 15.657 | 20.560 | 254.29 | 113.56 | 0 | 0 |
| dsum | 12.020 | 17.796 | 249.97 | 55.69 | 0 | 0 |
| IWSLT | 9.882 | 27.439 | 252.47 | 86.45 | 41 | 0 |

`role_marker_continuations` excludes a terminal `<|eot_id|>` because it is the expected shared stop token.

## Finish reasons

- cnn_dm_1000: legacy {'length': 99, 'stop': 1}; native {'stop': 100}
- dsum: legacy {'length': 97, 'stop': 3}; native {'stop': 100}
- IWSLT: legacy {'length': 97, 'stop': 3}; native {'stop': 100}

## Conclusion

On the identical fixed subset, the native Llama3 chat template improves all three primary metrics and eliminates the widespread length-capped behavior of the legacy prompt.
This supports prompt-template mismatch as a major explanation for Llama3.1's low *cross-model absolute* generation scores under the legacy/common PMPD protocol. It does not invalidate within-Llama3 compression comparisons, because those main results hold that legacy protocol fixed across BF16, uniform, and ours.
The native-template numbers are diagnostic only and must not replace the legacy/common main table.

## Fixed-subset check

- cnn_dm_1000: same question IDs and references across arms = True (100 samples).
- dsum: same question IDs and references across arms = True (100 samples).
- IWSLT: same question IDs and references across arms = True (100 samples).

## Paired examples

### cnn_dm_1000

- id 0; legacy: ` According to Amnesty International's annual report on the death penalty, the number of executions worldwide decreased by 22% in 2014, with at least 607 people executed globally. However, the report also notes a significant increase in the number of people sentenced to death, with at least 2,466 people handed the sentence last year, a 28% increase from 2013.`
  native: `Amnesty International's annual report on the death penalty has found that while the number of executions worldwide decreased by 22% in 2014, the number of people sentenced to death increased by 28%, with many countries using the death penalty as a tool to tackle terrorism and internal instability. The report highlights the use of the death penalty in countri`
- id 1; legacy: ` Here's a summary of the article: 

Bob Barker, the former host of "The Price Is Right," made a surprise return to the show after an eight-year absence. At 91, Barker looked spry and handled the "Lucky Seven" game with ease before turning over hosting duties to Drew Carey. The return marked a nostalgic moment for fans of the classic game show, which Barker h`
  native: `Bob Barker, a TV legend who hosted "The Price Is Right" for 35 years, made a surprise return to the show after an eight-year absence. At 91 years old, Barker looked spry and handled the classic "Lucky Seven" game with ease before passing the hosting duties to Drew Carey. Barker's return to the show was a nostalgic moment for fans, and he seemed to have picke`
- id 2; legacy: ` Here's a summary of the article: The article highlights some of the best videos of the week, featuring a range of topics including a tour of Mike Tyson's former mansion, a thrilling ride on the world's tallest and fastest roller coaster, and a dramatic interview with a spiritual leader accused of sexual assault. The videos also include a CNN exclusive on al`
  native: `A recent CNN article highlights some of the best videos of the week, featuring a range of topics including a tour of Mike Tyson's former mansion, the world's tallest and fastest roller coaster, and a dramatic interview with a spiritual leader accused of sexual assault. The article also includes a CNN exclusive on allegations against Bikram yoga creator Bikra`
### dsum

- id 0; legacy: ` The memo restricts all office communications to email and official memos, prohibiting the use of Instant Messaging during working hours, both internally and externally. Employees who violate this policy will face a warning, probation, and potential termination.  was not able to be generated.  was not able to be generated.  was not able to be generated.  was`
  native: `A new policy has been implemented restricting the use of Instant Messaging during working hours, both internally and externally, due to concerns over time waste. Employees who fail to comply will face a warning, probation, and potential termination for repeated offenses.`
- id 1; legacy: ` The memo restricts all office communications to email and official memos, prohibiting the use of Instant Messaging during working hours, both internally and externally. Employees who violate this policy will face a warning, probation, and potential termination.  was not able to be generated.  was not able to be generated.  was not able to be generated.  was`
  native: `A new policy has been implemented restricting the use of Instant Messaging during working hours, both internally and externally, due to concerns over time waste. Employees who fail to comply will face a warning, probation, and potential termination for repeated offenses.`
- id 2; legacy: ` The memo restricts all office communications to email and official memos, prohibiting the use of Instant Messaging during working hours, both internally and externally. Employees who violate this policy will face a warning, probation, and potential termination.  was not able to be generated.  was not able to be generated.  was not able to be generated.  was`
  native: `A new policy has been implemented restricting the use of Instant Messaging during working hours, both internally and externally, due to concerns over time waste. Employees who fail to comply will face a warning, probation, and potential termination for repeated offenses.`
### IWSLT

- id 6; legacy: ` Here is the translation of the text from French to English:

Since then, I have led around 70 design workshops around the world, with students, designers, and architects, and even the Technical Directors of Fortune 50 companies, and there is something about this exercise that reveals profound teachings on the nature of collaboration, and I would like to sha`
  native: `Here is the translation of the text from French to English:

Since then, I have led around 70 design workshops around the world, with students, designers, and architects, and even the Technical Directors of Fortune 50 companies, and there's something about this exercise that reveals profound lessons on the nature of collaboration, and I'd like to share some `
- id 27; legacy: ` Here is the translation of the text from French to English:

What kindergarten teams do differently is that they start with the marshmallow, and build successive prototypes, keeping the marshmallow always at the top, and thus have many opportunities to repair the prototypes in trouble. 

(Note: The text appears to be referring to a specific educational appr`
  native: `Here's the translation of the text from French to English:

What kindergarten teams do differently is that they start with the marshmallow, and build successive prototypes, keeping the marshmallow at the top, and thus have many opportunities to repair the prototypes in trouble.

Note: This text appears to be referring to the Marshmallow Challenge, a popular `
- id 119; legacy: ` Here is the translation of the text from French to English:

We are on the threshold of astonishing events in many fields. Yet, I think we must go back 300 years before the Age of Enlightenment to find a period when we fought against progress, when we fought these things more fiercely and on more fronts than we do now. 

Let me know if you need any further `
  native: `Here is the translation of the text from French to English:

We are on the threshold of astonishing events in many fields. Yet, I think we must go back 300 years before the Age of Enlightenment to find a period where we fought against progress, when we fought these things more fiercely and on more fronts than we do now.`
