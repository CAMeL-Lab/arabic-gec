from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2SeqGEC,
    MBartForConditionalGenerationGED,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed
)

import torch

class LMRewriter:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


    @classmethod
    def from_pretrained(cls, model_path, use_gpu):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available()
                              else 'cpu')

        model.eval()
        model = model.to(device)

        return cls(model, tokenizer)


    def rewrite(self, input_sentence):

        tokenized_sent = self.tokenizer(input_sentence, return_tensors="pt")

        gen_kwargs = {'num_beams': 5, 'max_length': 1024,
                      'num_return_sequences': 1}

        tokenized_sent = {k: v.to(self.model.device) for k, v in tokenized_sent.items()}

        generated_outputs = self.model.generate(tokenized_sent['input_ids'],
                                                **gen_kwargs)

        preds = self.tokenizer.batch_decode(generated_outputs,
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False
                                            )

        return preds[0]


# if __name__ == '__main__':
#     model_path = '/scratch/ba63/gec/models/gec/qalb14/bart'
#     rewriter = LMRewriter.from_pretrained(model_path, use_gpu=True)

#     sent = "سبحان الله الحكام العرب سيموت على الكرسي ليضهر أنه عنيد وقوي ، لوكان بشار يحب أرضه أو شعبه لخرج من الحكم شفقة ورحمة ببلد ضاع ، هنا زال "\
#             "قناع هذا الرئيس اللذي خيب ظن شعبه والشعوب المسلمة ، كل مال السورين نفق في شراء سلاح ليقتل به ، شتانا وحكام أوربا الذين يتركون الكرسي لم "\
#             "جرد فتنة بسيطة لحبهم لبلدهم"

#     rewriter.rewrite(sent)