from scripts.annotation.an_annotate_error_type import annotate
from scripts.alignment.al_align_input_reference import align_input_reference
from scripts.alignment.al_align_input_system import align_input_system
from scripts.evaluation.eval_functions import eval_multi_label_subclasses
import os


def process_align_annot_eval(ref_path, sys_path, output_path, uc):
    # System submission file
    exp_name = sys_path.split("/")[-1]

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # Aligned files
    out_align_input_ref = f"{output_path}/align_input_ref.tsv"
    out_align_input_sys = f"{output_path}/align_input_sys.tsv"

    # Annotated files
    out_annot_input_ref = f"{output_path}/annot_input_ref.tsv"
    out_annot_input_sys = f"{output_path}/annot_input_sys.tsv"

    # Alignment process
    print("Alignment in progress..")
    align_input_reference(ref_path, out_align_input_ref)
    align_input_system(sys_path, out_align_input_sys)

    # Annotation process
    print("Annotation in progress..")
    annotate(out_align_input_ref, out_annot_input_ref)
    annotate(out_align_input_sys, out_annot_input_sys)

    # Evaluation process
    print("Evaluation in progress..")
    eval_multi_label_subclasses(uc=uc, output_path=output_path,
                                extension=exp_name)



