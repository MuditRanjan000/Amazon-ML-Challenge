# LinkedIn Post Draft

Excited to share that our team has successfully completed the Amazon ML Challenge 2026, building a large-scale Business Entity Resolution system! 🚀

We tackled a massive dataset of over 12 million records filled with real-world noise—typos, missing addresses, and cross-lingual transliteration (English, Hindi, Tamil). After intense iteration, we achieved a final Macro-F0.5 score of **0.967**, placing us highly competitively on the leaderboard! 

Here are three key technical takeaways from our journey:

1️⃣ **Precision is King:** The F0.5 metric heavily penalizes false merges. We realized that blindly trying to maximize recall actually hurt our score. By implementing a custom decision layer that surgically dropped "lookalike" distractors in dense clusters, we found the winning edge.

2️⃣ **Infrastructure Scales Strategy:** We quickly hit memory limits trying to process the O(N²) search space. By migrating from Pandas to PyArrow Parquet and implementing memory-bounded sparse matrix operations (`sparse_dot_topn`) for our TF-IDF blocking, we were able to evaluate 346M candidate pairs efficiently on AWS without OOM errors.

3️⃣ **Rigorous Validation:** It's easy to fool yourself with data leakage. Enforcing a strictly frozen validation split with SHA-256 integrity checks kept us honest during feature engineering and prevented catastrophic overfitting. 

Huge shoutout to my amazing teammates Aayush and Ashank for their incredible work on candidate generation and modeling! This was a masterclass in modular pipeline design and teamwork. 

#MachineLearning #DataScience #AmazonMLChallenge #EntityResolution #AI #Python #DataEngineering
