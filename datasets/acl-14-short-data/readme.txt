
=====Description=====
As people tend to post comments for the celebrities, products, and companies, we use these keywords (such as ``\textit{bill gates}'', ``\textit{taylor swift}'', ``\textit{xbox}'', ``\textit{windows 7}'', ``\textit{google}'') to query the Twitter API. After obtaining the tweets, we manually annotate the sentiment labels (negative, neutral, positive) for these targets. In order to eliminate the effects of data imbalance problem, we randomly sample the tweets and make the data balanced. The negative, neutral, positive classes account for 25\%, 50\%, 25\%, respectively. Training data consists of 6,248 tweets, and testing data has 692 tweets.

=====Format=====
Each instance consists three lines:
- sentence (the target is replaced with $T$)
- target
- polarity label (0: neutral, 1:positive, -1:negative)

=====Citation=====
@inproceedings{dong2014adaptive,
  title={Adaptive Recursive Neural Network for Target-dependent Twitter Sentiment Classification},
  author={Dong, Li and Wei, Furu and Tan, Chuanqi and Tang, Duyu and Zhou, Ming and Xu, Ke},
  booktitle={The 52nd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2014},
  organization={ACL}
}

=====Contact=====
donglixp@gmail.com