Read through the results in the `experiments/runs-neuro` directory and the report in `analyze/neuro/report.html`.

Your job is to test the generalization of the build models to new subjects and new stories. To do this, take a handful of the best-performing non-redundant models (and a couple unique weak-performing models) from each of the runs and test them in two new settings:

1. New subjects: test the models on the same stories but with new subjects (original training was done on UTS03, now test on UTS01 and UTS02) with the same test stories. Produce plots of original performance vs new performance for each model, and a summary plot that shows all the models together.
2. New stories: test the models on the same subjects but with new stories (take all the training stories that were not originally used for training). Produce plots of original performance vs new performance for each model, and a summary plot that shows all the models together.

Write all your code and outputs for this into a subfolder here. It will help you to read the `experiments/evolve-neuro/src` folder, but do not edit anything there (you can copy and modify files into your subfolder if you want to).

Finally, include your results in a new report `analyze/neuro/report_generalization.html` that includes the results from the original runs and the new runs you did. Make sure to present the new runs separately from the original runs, and mention the differences in the report.
