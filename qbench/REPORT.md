# Quaternion Phase 2 Report

_Generated entirely from CSV and per-run metadata by `qbench.analysis.analyze`._

## Setup
TinyStories, data hash `f24c5b38b1bd631023408928e1533d0dc6064d6fad0257e2c1f8dab010eca2bd`. 2 layers, 4 heads, context 64, batch 8, vocab 1024. Final evaluation uses 2000000 sequential tokens with batch 64; curve evaluation uses 131072 tokens.

The awkward width-100 parameter-matched real control is intentionally omitted: matching it would require a dubious low-rank architecture and change more than algebra.

## Paired crossover statistics

| token_budget | n | mean_delta | ci_low | ci_high | signs | outcome |
| --- | --- | --- | --- | --- | --- | --- |
| 500000 | 5 | -0.0751156 | -0.119025 | -0.0312062 | -,-,-,-,- | quaternion win |
| 1000000 | 5 | -0.0346395 | -0.0571386 | -0.0121404 | -,-,-,-,- | quaternion win |
| 2000000 | 5 | -0.0269603 | -0.045443 | -0.00847755 | -,-,-,-,- | quaternion win |
| 5000000 | 5 | -0.00438605 | -0.0116289 | 0.00285678 | -,+,+,-,- | inconclusive |
| 10000000 | 5 | 0.0601886 | 0.0497146 | 0.0706626 | +,+,+,+,+ | real win |
| 25000000 | 3 | 0.0613949 | 0.0316966 | 0.0910932 | +,+,+ | real win |
| 50000000 | 3 | 0.037185 | 0.0127279 | 0.061642 | +,+,+ | real win |

## Ablations
| model | token_budget | n | mean_delta_vs_quat | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| quat_attn | 1000000 | 3 | 0.0260657 | -0.0101617 | 0.0622932 |
| quat_attn | 50000000 | 3 | -0.216499 | -0.248818 | -0.184181 |
| quat_ffn | 1000000 | 3 | -0.12195 | -0.143698 | -0.100202 |
| quat_ffn | 50000000 | 3 | -0.152132 | -0.165901 | -0.138364 |
| quat_gaussian | 1000000 | 3 | -0.0886047 | -0.162536 | -0.0146731 |
| quat_gaussian | 50000000 | 3 | -0.0286042 | -0.0534614 | -0.00374698 |
| quat_w64 | 1000000 | 3 | 0.31474 | 0.216601 | 0.412879 |
| quat_w64 | 50000000 | 3 | 0.267937 | 0.236498 | 0.299377 |

## Decision rule
H1 supported.

Crossover is reported only as a bracket between adjacent tested budgets whose mean/outcome changes sign; no interpolation.

Crossover bracket: 5000000 to 10000000 tokens

## Limitations
Single dataset, single small scale, and device-dependent throughput. Generation samples require completed 50M checkpoints.

## Generation samples

### quat, training seed quat-t50000000-s1-a63e1dc2afde
Prompt: `Once upon a time`

Once upon a time, there was a little bird named Fin. Billy had a while hea loved to run to play together. The bird was very happy.
One day, Buddy's friend, a squirrel came over to the talk and asked to his friend. "I will help you find a new friend, Cankers?" Bob thought of the bird was so happy. The bird said, "Thank you both!" So, the a shouted. He said, "I can be a big, here."
Tim was happy to take the kind friend, Sue. But the bird felt sad that he could

### quat, training seed quat-t50000000-s1-a63e1dc2afde
Prompt: `The little robot`

The little robot. The bird was tired. But before. The bird tried to hurt the bird. The bird was a bird became good friend. The bird had made a little cat. The bird and the bird were flew up.
Then, the bird found a big bird on the mak, and the the bird flew away. The bird was sad and said, "I am friend, Tim. I can help you." The bird said, "Ian welieve it was so happy and it was the bird! Let's get a warm bed."
Tim's friend, the bird. They became best friends. They played and took

### quat, training seed quat-t50000000-s1-a63e1dc2afde
Prompt: `One sunny morning`

One sunny morning, a small bird named Tim went to the park. He wanted to help, but he was playing with it was dark, but it was too fun.
After the bird flew down and smiled, and said, "Tom, Tim. Let's play together." Tim and Spot looked for a big tree. They played with the uncyard. They went to play in the park. They saw a small tree. Tim and Tim was very happy.
"Look, something unexpected happened? It is very amazing and small to play withe. The ball said, "Let's play with jungine

### quat, training seed quat-t50000000-s1-a63e1dc2afde
Prompt: `Mia found a strange`

Mia found a strange jungry tree. She said, "Go, I will come lazy! What is socome!" The cat felt the squirrel was so happy and done. They started to saft the big hat and said, "Of cabin! Let's go back home with a bag. I will play together!"
Tim did not be sad. He said, "You're else, bee happy. Let's go back to the park. You are not rest." Tim was surprised. He thanked their mom and said, "Thank you, mom. You are not good

### quat, training seed quat-t50000000-s2-87991e5b906d
Prompt: `Once upon a time`

Once upon a time, there was a little bird named Fin. Billy had a lot of fun. Bill loved to wild. They loved to play all day long.
One day, while Blue saw a big cat named Spot. Sally asked his friend, Doggy was playing with his friends. They wanted to help Tommy, but he wanted to get it. She put on the cat's machine and had a new friend shortant. He said, "Why are you sad. Please give me it!"
Lily was scared of the dog. The dog helped the cat, but it's

### quat, training seed quat-t50000000-s2-87991e5b906d
Prompt: `The little robot`

The little robot. He found a big box. The box was in the box and picked it up and saw a a small door. The bag was lucky and jumped into the hole. It was too high and bows. The dog was very big and laughed. The other bug was rounding.
The water was very happy. They enjoyed the ball home and played together. They had a great time, and they played together every day. The bird wet was happy and had fun.
<|endoftext|>
Once upon a time, there was a big cat. The cat named Lily and a cat, and her

### quat, training seed quat-t50000000-s2-87991e5b906d
Prompt: `One sunny morning`

One sunny morning, there was a three to the tasty ever. It was a big, red ball and it liked to play with.
One day, a little boy named Max saw a big box. He wanted to find a big yarn. He was very sad. He wanted to get out up. He decided to play with his toys and play in the park. Tim was playing under a big box. He liked to go outside and shout.
One day, Tim saw something unexpected happened. He was very amazing and had to play with his ball. He said, "Why can I was not rel

### quat, training seed quat-t50000000-s2-87991e5b906d
Prompt: `Mia found a strange`

Mia found a strange noise and saw a bigger. She was scared and wanted to help the law! She wanted to make the fruit, but it was too big! She knew it was the best!
Jack was very happy. He said, "Of cabin! Let's go back home!" The animals were happy and played in the forest. The other childrenor was not langed and else. It was happy and the esservy belive. They restized the other kids with the fill and they played together.
<|endoftext|>
Once upon a

### quat, training seed quat-t50000000-s3-477072b0f860
Prompt: `Once upon a time`

Once upon a time, there was a little bird named Fin was a wise cat named Spot. Max was very brave. He loved to play with his friends. Max loved to go to the park with his toy and help him.
One day, Sally came to his friend, Tim, saw a friend, a big squirrel named Canky. Max wanted to help Max. "I lost my ball!" So, he asked both played together.
Tim shouted to sleep on the ground. He felt scared, but he couldn't be smath. He said, "Can I play with me?" The bird said, "

### quat, training seed quat-t50000000-s3-477072b0f860
Prompt: `The little robot`

The little robot. Lea washed. He felt great in the tree and could not hurt.
<|endoftext|>
Once upon a time, in a small house, there was a wear. All the little little girl named Sue. Sue liked to hop on the mark, and rounds. She liked to preate things in the gift home. One day, they saw a big, there was a big banana. Tim thought he was a pencoling.
Tim thought, "Why are you sad?" The doll asked the cat said, "Yes, I am

### quat, training seed quat-t50000000-s3-477072b0f860
Prompt: `One sunny morning`

One sunny morning, it was a three. He liked to run ever after.
While he saw a little girl named Sally. She was so happy that he could make her gone. He said he would buy the bag. Lucy was very excited and happy, so she could not get it up. She decided to keep it out that she did not listen to her a susic.
At the end of the sunny day, Tim and the game became friends.
<|endoftext|>
Once upon a time, there was a boy named Tim. Tim liked to play outside with his toy cars. He liked to play

### quat, training seed quat-t50000000-s3-477072b0f860
Prompt: `Mia found a strange`

Mia found a strange noise and sweet. She said, "Yes, you can come lazy! Let's go to the tree!" The sun was the top of the stepped, but they started to saft it.
The mix, the tree saw an idea. "That's fun!" The animals were happy and happy. They thanked their friends.
<|endoftext|>
Once upon a time, there was a little bird named Tim. Tim was a movele. Tim liked to play with his friends. One day, Tim went to their mom and Sam asked. They was so happy.
Tim was good

### real, training seed real-t50000000-s1-ad37ca439ff7
Prompt: `Once upon a time`

Once upon a time, there was a little bird named Fin. Billy had a while hea loved to run to play together. One day, Billy met a small bird named Buddy. Blue wanted to help my mail. Sally asked to help the bird, "What are you sad?" So, "I can help me?" The bird said, "I am him. Bunny!" So, Bob and both played together.
The shaper came to find it. The bird felt better. They watched the bird and the bird played together all of the tree. The bird made the bird and the bird's

### real, training seed real-t50000000-s1-ad37ca439ff7
Prompt: `The little robot`

The little robot shouted, "I want to see the in the tree?" Le smiled and said, "We can't keep you so creatures. We making welcome away."
The window put the stick drink in the tree. Suddenly, the the man was closet and preate. Instevoboke the ladder, a boat came to the banana. A wise man saw the water was full of dark, and an adventure. The boat was so happy.
Then, something unexpected happened, a big

### real, training seed real-t50000000-s1-ad37ca439ff7
Prompt: `One sunny morning`

One sunny morning, a small bird named Tim went to the park. He wanted to swim and play together. But it was too small, but it was too too small. He was too angry. He wanted to help her. So, he walked to the park and walked, so he saw a big hole up. He decided to eat it. So, he explore it was a big plan.
At the park, Tim saw a man and said, "Thank you, Tom! They are sad, but they am a friends. He wanted to go for the hourse to the park. They ran to the park and

### real, training seed real-t50000000-s1-ad37ca439ff7
Prompt: `Mia found a strange`

Mia found a strange phone. They both wanted to go home and play with the r lawn was very pretty.
"Come on the slide, Mom! I'm gloomes! "Df my cars," said Max. "I love you. I can be careful."
"Okay, Lily, I will be very very good. I don't be nice. You need to lake things and elate."
They hoper, they came back to the park. They were sad and wanted to be!"
Mom and Ben said, "Okay, mom. You are not a good dog.

### real, training seed real-t50000000-s2-ee46012d60c3
Prompt: `Once upon a time`

Once upon a time, there was a little bird named Fin. Billy had a while whoa loved to run to play together. One day, Billy met a small bird named Buddy, so she decided to play with her friends. The bird stayed in the sky and laughed.
One sunny day, the bird saw aket pitch. It was very happy. The girl had a big to be a both of food. The bird shy and the cat played in the sky. The colorful bird watched the bird and the bird went to the gentle. The bird had red ball and learned that the

### real, training seed real-t50000000-s2-ee46012d60c3
Prompt: `The little robot`

The little robot was scared and scared.
So, the little fish saw the fish, but the fish was a bird too close. The man felt happy. It jumped up and joke up. He flew up and laughed and found a fun stick. The boat laughed and jumped on the tree. Bob decided to play with the big pepper.
<|endoftext|>

Once upon a time, there was a boy named Tim. He loved to play with his woods. He had a lot of fun. He would hop to race.
One day, Tim went to the park. He wanted to play and have

### real, training seed real-t50000000-s2-ee46012d60c3
Prompt: `One sunny morning`

One sunny morning, a small bird named Tim went to the park. He wanted to help his friends.
Tim and Sue liked to play with the bird. They would play with the bird, and it were happy. They became best friends and has funny. Tim was happy, and they were very happy.
<|endoftext|>
Once upon a time, there was a little girl named Sue. Sue loved to play outside. One day, Sue saw a big smile in the park. One day, Sue came to play with her car, but the small boy could note. The toy said, "Let's play with jungine

### real, training seed real-t50000000-s2-ee46012d60c3
Prompt: `Mia found a strange`

Mia found a strange noise and sweet. She said, "Yes, let's come lazy! Let's play with your books!" She felt the bookster and thanked them.
<|endoftext|>
Once upon a time, there was an enother dog named Tom. Max loved to play with his bed. They would run and run in the car. One day, Max found a big, big, land. Max wanted to play outside to play with the ball. He said, "Can you help me go help!" Max was surprised. He had a new toy. He started his paper.
Tim was good

### real, training seed real-t50000000-s3-fc6a49d25c97
Prompt: `Once upon a time`

Once upon a time, there was a little bird named Fin. Bob loved to play with his friends. One day, he found a small bird. He saw a big, pretty but the bird was too far away. It asked, "What is a big dog?" The bird was surprised because the bird said, "Yes, but I still mean to be out of the squirrel. Let's play with the goose to find a both!" So, the bird shouted. Tim and the bird nodded, and said, "You can't play it!" They played together all day.
Tim and the farme is scared and happy

### real, training seed real-t50000000-s3-fc6a49d25c97
Prompt: `The little robot`

The little robot said, "Torry, cannot in the garden and go back. You can be a good time, a barage that made a police mush! I must be a little girl named Daddy. Can you see that other friends can ruinules.
<|endoftext|>
One day, a loud friend, Tim. Tim found a big tree in the forest. He had a toy car. Tim woods to help it. Tim wanted to see the car.
Tim saw a big big, pretty bug. He asked, "Why are you sad?"

### real, training seed real-t50000000-s3-fc6a49d25c97
Prompt: `One sunny morning`

One sunny morning, a small bird named Tim went to the park. He saw a big, big tree on the tree. Tim looked at the garden and saw a cat. The cat was sad and wanted to help him.
The cat said, "Let's play with you, I will help you get fix the ball. The cat walked away with the dog." Tim felt sad and said, "I don't hurt you, Tim. I will shray down!"
But then, something unexpected happened. The big cat picked up the ball back to its mroble. Tim was so happy and said, "Why are

### real, training seed real-t50000000-s3-fc6a49d25c97
Prompt: `Mia found a strange`

Mia found a strange car and was so happy.
<|endoftext|>
Once upon a time, there was a little girl named Mia. She was a girl named Sue. She liked to driver. She would saft it up and sign. One day, she saw a small tree in the park with a small tree. The cat wanted to help her friends.
Sue was so happy to see who wanted to help her mommy. She said, "Stophhe, I can not find some rest." They all played together with the fill and had done. They never got lost their toys, good

