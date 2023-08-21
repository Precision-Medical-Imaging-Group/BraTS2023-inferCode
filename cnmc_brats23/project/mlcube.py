"""MLCube handler file"""
import os 
os.environ["MKL_SERVICE_FORCE_INTEL"] = '1'  # see issue #152
import typer
import os

app = typer.Typer()


@app.command("infer")
def infer(
    data_path: str = typer.Option(..., "--data_path"),
    output_path: str = typer.Option(..., "--output_path"),
    challenge: str = typer.Option(..., "--challenge"),
):
    
    if (challenge in ["ped", "PED", "BraTS2023-PED"]):
        from ped_runner import setup_model_weights, batch_processor
    elif (challenge in ["men", "MEN", "BraTS2023-MEN"]):
        from men_runner import setup_model_weights, batch_processor
    elif (challenge in ["met", "MET", "BraTS2023-MET"]):
        from met_runner import setup_model_weights, batch_processor
    else:
        raise Exception("Please, input a valid challenge name:\n"
                        "Supported challenges are:\n"
                        "- 'ped' or 'PED' or 'BraTS2023-PED'\n"
                        "- 'men' or 'MEN' or 'BraTS2023-MEN'\n"
                        "- 'met' or 'MET' or 'BraTS2023-MET'")

    setup_model_weights()
    batch_processor(data_path, output_path)

    return output_path



@app.command("hotfix")
def hotfix():
    # NOOP command for typer to behave correctly. DO NOT REMOVE OR MODIFY
    pass


if __name__ == "__main__":
    app()
