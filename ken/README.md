# Transformer Binary Classifier Model
## Ken Kliesner
Implemented using HuggingFace Transformer Pre-Trained Model (Based on BERT, PyTorch, and TensorFlow):
    https://huggingface.co/docs/transformers/tasks/sequence_classification

## Primary Files:
- transformer_bi_label.ipynb : contains all of the main functioning code for the model
- data/ : has all of the relevant data
- everything else : has test files and other experimented with files

### Overview:
Text classification is a common NLP task that assigns a label or class to text. Some of the largest companies run text classification in production for a wide range of practical applications. One of the most popular forms of text classification is sentiment analysis, which assigns a label like 🙂 positive, 🙁 negative, or 😐 neutral to a sequence of text.

### Procedure:
- Finetuning DistilBERT on the IMDb dataset to determine whether a movie review is positive or negative.
- Using my own fine-tuned model for inference.
- The task illustrated in this tutorial is supported by the following model architectures:
    ALBERT, BART, BERT, BigBird, BigBird-Pegasus, BioGpt, BLOOM, CamemBERT, CANINE, ConvBERT, CTRL, Data2VecText, DeBERTa, DeBERTa-v2, DistilBERT, ELECTRA, ERNIE, ErnieM, ESM, FlauBERT, FNet, Funnel Transformer, GPT-Sw3, OpenAI GPT-2, GPTBigCode, GPT Neo, GPT NeoX, GPT-J, I-BERT, LayoutLM, LayoutLMv2, LayoutLMv3, LED, LiLT, LLaMA, Longformer, LUKE, MarkupLM, mBART, MEGA, Megatron-BERT, MobileBERT, MPNet, MVP, Nezha, Nyströmformer, OpenLlama, OpenAI GPT, OPT, Perceiver, PLBart, QDQBert, Reformer, RemBERT, RoBERTa, RoBERTa-PreLayerNorm, RoCBert, RoFormer, SqueezeBERT, TAPAS, Transformer-XL, XLM, XLM-RoBERTa, XLM-RoBERTa-XL, XLNet, X-MOD, YOSO


### Performance:
This model performed fairly well as pre-trained model (that was pre-trained on IMDB movie review data, which is fairly similar to Wikipedia review data).

    Epoch	Training Loss	Validation Loss	    Accuracy
    1	    0.231200	    0.193241	        0.926120
    2	    0.151500	    0.234670	        0.929600

That said, the model was too large to apply to our Wikipedia data, and I ran into issues applying the classifier to our test dataset, even on the Google Colab GPU.  In the future this would be better applied using slurm jobs on the university cluster GPU nodes, or on a larger GPU web service like AWS.