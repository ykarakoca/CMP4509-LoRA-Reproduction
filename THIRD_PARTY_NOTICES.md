# Attribution and publication scope

Reviewed 16 November 2026. These are source observations, not a legal opinion.

- RoBERTa model card identifies the model license as MIT: https://huggingface.co/FacebookAI/roberta-base/blob/main/README.md . Model/tokenizer assets are downloaded upstream rather than redistributed here.
- The GLUE dataset card labels its umbrella license `other`: https://huggingface.co/datasets/nyu-mll/glue . The separate Stanford SST-2 card labels its license `unknown`: https://huggingface.co/datasets/stanfordnlp/sst2/blob/main/README.md . Do not infer a permissive dataset license from a software toolkit's license. Raw sentence text is not included in this repository; the loader retrieves it upstream. The repository contains IDs, measured predictions and evaluation annotations for auditing.
- Cite the original SST paper and respect upstream terms when downloading or using data: https://nlp.stanford.edu/sentiment/ and https://aclanthology.org/D13-1170/ . Dataset redistribution permission is not asserted here.
- Custom LoRA code and the experimental runtime were supplied as the team's project implementation. This review has not independently established the provenance of every original line. No external LoRA implementation is vendored; the method is attributed to Hu et al.
- Dependencies are installed separately and retain their own terms. The package imports PyTorch, Hugging Face Transformers/Datasets, NumPy and Matplotlib; scikit-learn is recorded for the original split environment. Their source is not bundled.
- The course announcement, lecture slides, original paper PDF and instructor's LaTeX template are not redistributed in this code package. The final report will use the supplied template separately.
- No LICENSE file has been fabricated. The three authors should choose a license for their own code after confirming ownership and attribution. Public availability alone does not make the work open-source licensed.
