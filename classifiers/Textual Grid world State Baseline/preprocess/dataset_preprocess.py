def get_tensor_dataset(df, tokenizer):

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    token_type_ids = []
    attention_masks = []
    labels = []
    topic_ids = []

    for count, item in tqdm(
        enumerate(
            zip(
                df["GameId"],
                df["bylevel_color_context"],  # df["nonspatial_color_context"],
                df["InputInstruction"],
                df["IsInstructionClear"],
            )
        ),
        total=len(df),
        desc="Tokenizing data",
    ):
        z, w, x, y = item
        encoded_dict = tokenizer.encode_plus(
            w,
            x,
            add_special_tokens=True,
            max_length=max_seq_length,
            padding="max_length",  # use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or use `padding='max_length'`
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids.append(encoded_dict["input_ids"])

        if "token_type_ids" in encoded_dict:
            token_type_ids.append(encoded_dict["token_type_ids"])

        attention_masks.append(encoded_dict["attention_mask"])
        labels.append(y)

        topic_ids.append(z)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    return TensorDataset(input_ids, attention_masks, labels)
