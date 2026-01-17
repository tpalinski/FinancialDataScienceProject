import typer
from enum import Enum

from model.model import fine_tune_model, train_basic
from model.utils import load_data
from model.visualize import visualize_dataset, simulate_model
from model.explain import explain_model

app = typer.Typer(help="Simulation CLI")

class Index(str, Enum):
    ETH = "ETH"
    BTC = "BTC"
    XRP = "XRP"
    LTC = "LTC"
    XLM = "XLM"

@app.command()
def simulate_single(
        index: Index = Index.ETH, 
        output_shap: bool = False, 
        output_visualize: bool = False,
        test_days: int = 200, 
        largest_window: int = 50, 
        validation_days: int = 100,
        epochs: int = 300,
        buy_prob: float = 0.5,
        sell_prob: float = 0.6):
    data = load_data()
    current = data[index.value]
    train = current[:-test_days]
    test = current[-(test_days+largest_window):]
    if output_visualize:
        visualize_dataset(current)
    model = train_basic(train, validation_days, epochs)
    simulate_model(test, model, largest_window, buy_prob, sell_prob, title=f"{index.value} model")
    if output_shap:
        explain_model(model, test, f"{index.value} model")

@app.command()
def simulate_multidata(
        index: Index = Index.ETH,
        validation_days: int = 100,
        test_days: int = 200, 
        largest_window: int = 50, 
        epochs: int = 300,
        buy_prob: float = 0.5,
        sell_prob: float = 0.6
        ):
    data = load_data()
    current = data[index.value]
    train = current[:-test_days]
    test = current[-(test_days+largest_window):]
    model = train_basic(train, validation_days, epochs)
    simulate_model(test, model, largest_window, buy_prob, sell_prob, title=f"{index.value} model")
    for key in data.keys():
        if key != index.value:
            test_new = data[key][-(test_days+largest_window):]
            simulate_model(test_new, model, largest_window, buy_prob, sell_prob, title=f"{index.value} model, {key} data")

@app.command()
def simulate_finetuned(
        index: Index = Index.ETH,
        output_shap: bool = False,
        validation_days: int = 100,
        test_days: int = 200, 
        largest_window: int = 50, 
        epochs: int = 300,
        buy_prob: float = 0.5,
        sell_prob: float = 0.6,
        fine_tune_epochs = 100
        ):
    data = load_data()
    current = data[index.value]
    train = current[:-test_days]
    test = current[-(test_days+largest_window):]
    model = train_basic(train, validation_days, epochs)
    simulate_model(test, model, largest_window, buy_prob, sell_prob, title=f"{index.value} model")
    for key in data.keys():
        if key != index.value:
            train_new = data[key][-(test_days+largest_window):]
            test_new = data[key][-(test_days+largest_window):]
            fine_tuned = fine_tune_model(model, train_new, validation_days, fine_tune_epochs)
            scratch = train_basic(train_new, validation_days, epochs)
            simulate_model(test_new, model, largest_window, buy_prob, sell_prob, title=f"{index.value} model, {key} data")
            simulate_model(test_new, fine_tuned, largest_window, buy_prob, sell_prob, title=f"{index.value} model, finetuned on {key} data")
            simulate_model(test_new, scratch, largest_window, buy_prob, sell_prob, title=f"{key} model, {key} data")
            if output_shap:
                explain_model(model, test_new, f"{key}, {index.value} model")
                explain_model(fine_tuned, test_new, f"{key}, {index.value} + {key} model")
                explain_model(scratch, test_new, f"{key}, {key} model")

if __name__ == '__main__':
    app()
