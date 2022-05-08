import os
import torch

def save_model(args, model, tokenizer, logger, save_mode="final"):
    output_dir = os.path.join(args.output_dir, save_mode)
    if args.local_rank in [-1, 0]:  # Save the final model checkpoint
        if not os.path.exists(output_dir) and args.local_rank in [-1, 0]: os.makedirs(output_dir)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`. They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))