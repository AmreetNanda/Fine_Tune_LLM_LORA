import math

# Compute perplexity from total loss and toekn count
def perplexity(loss, num_tokens):
    return math.exp(loss/num_tokens)
