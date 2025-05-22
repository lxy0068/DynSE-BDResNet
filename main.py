import argparse
import os

from dbres import calculate_dbres_scores 
from data_splits import stratified_test_vali_split
from train_resnet import run_model_training
from xgboost_integration import calculate_xgboost_integration_scores

def main(
    data_directory,
    stratified_directory,
    test_size,
    vali_size,
    cv,
    recalc_features,
    spectrogram_directory,
    model_name,
    recalc_output,
    dbres_output_directory,
    bayesian
):
    # Define stratified feature columns
    stratified_features = ["Normal", "Abnormal", "Absent", "Present", "Unknown"]

    # Data segmentation path
    split_subdir = "cv_False" if not cv else f"cv_{cv}_stratified_{stratified_cv}"
    cnum = "seed_42"
    base_split_path = os.path.join(stratified_directory, split_subdir, cnum)

    # Training/validation/testing data paths
    train_data_directory = os.path.join(base_split_path, "train_data")
    vali_data_directory = os.path.join(base_split_path, "vali_data")
    test_data_directory = os.path.join(base_split_path, "test_data")

    # Perform data segmentation
    stratified_test_vali_split(
        stratified_features=stratified_features,
        data_directory=data_directory,
        stratified_directory=stratified_directory,
        test_size=test_size,
        vali_size=vali_size,
        cv=cv,
        random_states=[42],
    )

    run_model_training(
        recalc_features,
        train_data_directory,
        vali_data_directory,
        spectrogram_directory,
        model_name,
        "BinaryPresent",
        "data/models",
        "binary_present",
        bayesian,
        None,
    )

    run_model_training(
        recalc_features,
        train_data_directory,
        vali_data_directory,
        spectrogram_directory,
        model_name,
        "BinaryUnknown",
        "data/models",
        "binary_unknown",
        bayesian,
        None,
    )

    dbres_scores = calculate_dbres_scores(
        recalc_output,
        test_data_directory,
        dbres_output_directory,
        "data/models/model_BinaryPresent.pth",
        "data/models/model_BinaryUnknown.pth",
    )

    xgb_scores = calculate_xgboost_integration_scores(
        train_data_directory,
        test_data_directory,
        dbres_output_directory,
        "data/models/model_BinaryPresent.pth",
        "data/models/model_BinaryUnknown.pth",
        bayesian=bayesian
    )

    return dbres_scores, xgb_scores


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="DBResAndXGBoostIntegration")
    parser.add_argument(
        "--data_directory",
        type=str,
        help="The directory containing all of the data.",
        default=r"C:\Users\ftlxy\Downloads\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data",
    )
    parser.add_argument(
        "--stratified_directory",
        type=str,
        help="The directory to store the split data.",
        default="data/stratified_data",
    )
    parser.add_argument(
        "--vali_size", type=float, default=0.16, help="The size of the test split."
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="The size of the test split."
    )
    parser.add_argument(
        "--cv", type=bool, default=False, help="Whether to run cv."
    )
    parser.add_argument(
        "--recalc_features",
        action="store_true",
        help="Whether or not to recalculate the log mel spectrograms used as "
        "input to the ResNet.",
    )
    parser.add_argument(
        "--no-recalc_features", dest="recalc_features", action="store_false"
    )
    parser.set_defaults(recalc_features=True)
    parser.add_argument(
        "--spectrogram_directory",
        type=str,
        help="The directory in which to save the spectrogram training data.",
        default="data/spectrograms",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The ResNet to train. Current options are resnet50 or resnet50dropout.",
        choices=["resnet50", "resnet50dropout"],
        default="resnet50dropout",
    )
    parser.add_argument(
        "--recalc_output",
        action="store_true",
        help="Whether or not to recalculate the output from DBRes.",
    )
    parser.add_argument(
        "--no-recalc_output", dest="recalc_output", action="store_false"
    )
    parser.set_defaults(recalc_output=True)
    parser.add_argument(
        "--dbres_output_directory",
        type=str,
        help="The directory in which DBRes's output is saved.",
        default="data/dbres_outputs",
    )
    parser.add_argument(
        '--disable-bayesian', 
        dest='bayesian', 
        action='store_false', 
        default=True,
        help='Disable Bayesian features (default: Bayesian is enabled)'
    )

    args = parser.parse_args()

    if "dropout" in args.model_name:
        args.bayesian = True
    else:
        args.bayesian = False

    dbres_scores, xgb_scores = main(**vars(args))
