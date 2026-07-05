"""LoRA fine-tuning experiment: chunk-category classification.

The corpus documents carry authoritative categories in ``data/eval/SOURCES.md``
(legal / academic / technical / security / finance), which gives the experiment
free, honest ground-truth labels. The task: classify a text chunk into its
document category — useful for routing domain presets, and small enough to
train in minutes while still demonstrating the full PEFT/LoRA + tracked
comparison workflow end to end.
"""
