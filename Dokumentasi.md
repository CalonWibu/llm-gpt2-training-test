# DOKSLI llm-gpt2-training-test

jadi disini saya mencoba untuk train model ai yang sudah ada yaitu pakai gpt2, saya pakai model ai ini karena kendala device yang low budget.
pada project kali ini saya mencoba menggunakan beberapa data untuk melatih ai dengan method fine-tunning
dan sebelumnya saya juga mencoba membuat sebuah model ai sendiri dari 0 dari Transformer, namun mungkin karena data yang di gunakan terlalu sedikit jadi ainya rada abnormal.

! Saya hanya mengunggah dokumentasi karena model ai terlalu besar dan tidak kuat untuk di push di github.

jadi saya menggunakan sekitar 10000 data namun karena proses trainingnya yang sangad sangad lama dan keburu laptop saya duar duar jadi saya kurangi jadi 100 - 500 data dan batch 20x.

contoh data yang saya pakai untuk finetunning kali ini seperti
```
{"text": "A: Katanya sih stasiun itu murah!\nB: Aku sedang berenang sekarang!\nA: Jangan menulis terus, nanti ramai!\nB: Jangan berbelanja terus, nanti lambat..\nA: Aku sering belajar di pantai dekat sini!\nB: Wah, kamu bisa berkumpul juga?\nA: Kok bisa sih lambat banget gini?\nB: Aku dulu pernah memasak sampai dingin..\nA: Aku baru bermain, jadi masih cepat!\nB: Aduh, aku lambat nih."}
```

dan contoh code untuk training

```
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

def main():
    # Load tokenizer dan model GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))

    # Load dataset dari CSV
    dataset = load_dataset('csv', data_files={'train': 'indonesian_chat.csv'})

    # Tokenisasi isi kolom 'chat' (bukan 'text')
    def tokenize_function(examples):
        return tokenizer(examples['chat'], truncation=True, max_length=128, padding="max_length")

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["id", "chat", "label"])
    # Data collator untuk causal language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training args
    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned-game",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )

    print("Mulai fine-tuning...")
    trainer.train()
    print("Fine-tuning selesai!")

    # Simpan hasil
    model.save_pretrained("./gpt2-finetuned-game")
    tokenizer.save_pretrained("./gpt2-finetuned-game")

if __name__ == "__main__":
    main()
```

sebenernya pakai data kayak gimanapun juga ga masalah atau mau langsung percakapan atau obrolan di whatsapp. karena method ini adalah memberi data ke ai yang sudah di latih namun masih cacad.

dari hasil percobaan pertama - ke 8 yang hanya menambah batch di setiap percobaan dan hasilnya jawaban si ai kayak copas dari datanya langsung contoh

prompt
```
siapa presiden pertama indonesia
```

jawaban
```
soekarno
```

dan pada data juga sama persis yaitu
```
("Siapa presiden pertama Indonesia?", "Soekarno")
```
jadi ai itu terkesan hanya copas jawaban dari data, bukan mempelajari data.
namun karena aku adalah manusia yang baikhati suka membantu dan rajin menabung akhirnya aku coba mengurangi batch jadi 5x dan hasilnya cacad juga.

prompt
```
siapa presiden pertama indonesia
```

jawaban
```
soekaswindow
```
jadi dari ini aku bisa simpulkan yaitu:

- pertama data yang aku beri itu sangat sangat sangat sangat sangat sangat sangat sangat sangat sangat sangat sangat sangat sangat sangat sedikit di banding model ai lain yang punya sampai miliar parameter, hal ini menyebabkan si model kecerdasannya juga sangat sangat sangat sangat sangat sangat sangat sangat sangat sedikit atau bloon.

- kedua masalah batch atau berapa kali si model ai di ajarin datanya, jika kebanyakan si ai akan menjadi copas atau plug and play, dan jika terlalu sedikit maka ai akan ga jelas.

- ke 3 adalah masalah duit, karena duit yang terbatas jadi pada project kali ini saya hanya mengandalkan laptop yang tengah tengah kebawah. intinya minimal ram di atas 64gb dan storage juga butuh yang gedhe.

- ke 4 adalah masalah waktu. karena ketika saya memulai project ini waktu yang saya miliki cukup sedikit karena banyak urusan, misal di suruh jemur baju dll......

- ke 5 adalah masalah internet, jujur internet ini wajib dan bahkan saya bisa menghabiskan 10gb dalam waktu kurang dari 20 menit hanya untuk pull model ai.

okeh sekian terimakajih ❤️😁👌
