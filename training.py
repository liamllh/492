import datetime
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from torch.nn import LSTM, Linear, Module, MSELoss, L1Loss
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from math import log
from typing import Any, Literal, Optional
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import pickle

from directories import derivatives_paths_run
from make_files import GJF
from parser import parse_log
from grokfast import gradfilter_ma, gradfilter_ema


plt.style.use('dark_background')


def iterate_over_derivatives():
	"""
	Yields a derivative and its corresponding job log file as 2-tuple of Paths for each PAH class of derivatives
	:return:
	"""
	for derivative_paths in derivatives_paths_run:
		this_derivative_paths = sorted(Path.iterdir(Path(derivative_paths)))
		gjfs = this_derivative_paths[::2]
		logs = this_derivative_paths[1::2]
		assert len(gjfs) == len(logs), f"Found different number of job and log files ({len(gjfs)} and {len(logs)})."

		for i, job in enumerate(gjfs):
			log_ = logs[i]
			yield str(job), str(log_)

	return


def pad_square_tensor_zeros(tensor_in: Tensor, padding: int) -> Tensor:

	shape = tensor_in.shape
	assert len(shape) == 2, f"Expected square Tensor, got {len(shape)} dimensions!"
	assert shape[0] == shape[1], f"Expected square Tensor, got side lengths {shape[0]} and {shape[1]}!"

	side_length = shape[0]
	padding_needed = padding - side_length
	top_padding = torch.zeros((padding_needed, side_length)).to(tensor_in.device)
	right_padding = torch.zeros((padding, padding_needed)).to(tensor_in.device)

	tensor_top_padded = torch.vstack((tensor_in, top_padding))
	tensor_padded = torch.hstack((tensor_top_padded, right_padding))

	return tensor_padded


def organize_training_data(n: int | None = None, reorder: bool = False) -> tuple[tuple[Tensor, Tensor], Tensor]:
	"""
	Returns padded connection and distance matrices, and final SCF energies, for each gjf in any derivative folder
	:return: (connection matrices, distance matrices), final relaxation energies
	"""

	max_n_atoms = 0
	cx_matrices: list[Tensor] = list()
	dx_matrices: list[Tensor] = list()
	matrix_aligned_final_energies: list[float] = list()
	current_iter = 0

	def break_loop(curr_iter: int) -> bool:
		if n is not None:
			if curr_iter >= n:
				return True
		return False

	for d, l in tqdm(iterate_over_derivatives(), desc="Generating characteristic matrices"):

		gjf = GJF(d)
		# record the congruent derived connection and distance matrices with same energy for extra training data at no cost
		congruent_gjfs = gjf.generate_congruent_isomers()
		energies, cycles, time = parse_log(l)
		# update minimum required padding
		max_n_atoms = max(len(gjf.atom_dict.keys()), max_n_atoms)

		# skip files with abnormal termination
		if len(energies) == 0:
			continue

		# record last energy
		final_energy = energies[-1]

		# reorder
		if reorder:
			gjf.reorder_c_first()

		# generate data
		cx_matrices.append(Tensor(gjf.get_connection_matrix()))
		dx_matrices.append(Tensor(gjf.get_normalized_distance_matrix()))
		matrix_aligned_final_energies.append(final_energy)

		current_iter += 1
		if break_loop(current_iter):
			break

		for congruent_gjf in congruent_gjfs:

			if reorder:
				congruent_gjf.reorder_c_first()

			cx_matrices.append(Tensor(congruent_gjf.get_connection_matrix()))
			dx_matrices.append(Tensor(congruent_gjf.get_normalized_distance_matrix()))
			matrix_aligned_final_energies.append(final_energy)

	padded_cx_tensors, padded_dx_tensors = list(), list()
	for i, cx_matrix in enumerate(cx_matrices):

		cx_matrix = pad_square_tensor_zeros(cx_matrix, max_n_atoms)
		dx_matrix = pad_square_tensor_zeros(dx_matrices[i], max_n_atoms)

		padded_cx_tensors.append(cx_matrix)
		padded_dx_tensors.append(dx_matrix)

	cx_matrices_out = torch.stack(padded_cx_tensors, dim=0)
	dx_matrices_out = torch.stack(padded_dx_tensors, dim=0)
	energies_out = Tensor(matrix_aligned_final_energies)

	return (cx_matrices_out, dx_matrices_out), energies_out


class ChemLSTM(Module):

	def __init__(
			self,
			input_size: int,
			hidden_size: int,
			output_size: int,
			num_layers: int = 2,
			dropout: float = 0.15,
			device: torch.device = torch.device("cuda:0")
	):

		super(ChemLSTM, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_layers = num_layers
		self.dropout = dropout
		self.fc_layer = Linear(hidden_size, output_size, device=device)
		self.device = device
		self.lstm = LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout,	device=device)

	def init_hidden(self, batch_size: int) -> tuple[Tensor, Tensor]:

		if batch_size == 1:
			h0 = torch.randn(self.lstm.num_layers, self.hidden_size).to(self.device)
			c0 = torch.randn(self.lstm.num_layers, self.hidden_size).to(self.device)
		else:
			h0 = torch.randn(self.lstm.num_layers, batch_size, self.hidden_size).to(self.device)
			c0 = torch.randn(self.lstm.num_layers, batch_size, self.hidden_size).to(self.device)

		return h0, c0

	def forward(
			self,
			inputs: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:

		# extract features
		input_shape = inputs.shape
		ndim = len(input_shape)

		# batched
		if ndim == 3:
			batch_size, input_size, _ = input_shape
		# unbatched
		elif ndim == 2:
			batch_size = 1
			input_size, _ = input_shape

		else:
			raise ValueError(f"Unexpected number of dimensions; expected 2 (unbatched) or 3 (batched); got {ndim}.")
		hidden_states, cell_states = self.init_hidden(batch_size)

		# do forward pass
		out, (hidden_states, cell_states) = self.lstm(inputs, (hidden_states, cell_states))
		# pass final hidden to fully connected layer
		# out = torch.sigmoid(self.fc_layer(hx[-1]))
		out = self.fc_layer(hidden_states[-1])

		return out, (hidden_states, cell_states)

	def count_params(self) -> int:

		count: int = 0
		for layer in self.parameters():
			count += layer.numel()

		return count


def run_grokfast(
		grads: dict[str: Tensor] | None,
		model: ChemLSTM,
		func: Literal["ma", "ema"],
		alpha: float,
		window_size: int,
		lamb: float
) -> dict[str: Tensor]:

	if func == "ma":
		grads_out = gradfilter_ma(model, grads, window_size, lamb)
	else:
		grads_out = gradfilter_ema(model, grads, alpha, lamb)

	return grads_out


def run_lr_atten(init_lr: float, epoch_fraction: float, atten_power: float = 0.0, atten_end: float = 0.0,) -> float:
	# print(f"{init_lr = }, {epoch_fraction = }, {atten_power = }, {atten_end = }")
	curr_lr = lr_atten_parabolic(init_lr, atten_end, epoch_fraction, atten_power)
	return curr_lr


def sma(series: list[float], period: int) -> float:
	num = len(series)
	if num > period:
		return sum(series[-period:]) / period
	else:
		return sum(series) / num


def train_model_verbose(
		model: ChemLSTM | str,
		train_loader: DataLoader,
		test_loader: DataLoader,
		epochs: int = 32,
		loss_functional: torch.nn.modules.loss = MSELoss,
		optimizer_func: torch.optim = AdamW,
		lr: float = 0.01,
		do_grokfast: bool = False,
		grokfast_kwargs: Optional[dict[str: Any]] = None,
		do_lr_atten: bool = False,
		lr_atten_kwargs: Optional[dict[str: Any]] = None,
		ma_period: int = 30,
		save: bool = False,
		save_path: Optional[str] = None,
		update_freq: int = 8,
) -> None:
	"""
	:param model: model to train
		if type(model) is ChemLSTM: this model will be trained
		if type(model) is str: model will be loaded from the child directory in ./pretrained_models/{model}
	:param train_loader: train data DataLoader object
	:param test_loader: test data DataLoader object
	:param epochs: number of epochs to train
	:param loss_functional: torch.nn function for computing loss
	:param optimizer_func: torch.nn function for gradient descent
	:param lr: learn rate, or initial learn rate if do_lr_atten is True
	:param do_grokfast: whether to use gradient accumulation
	:param grokfast_kwargs: {"func": Literal["ma", "ema"], "alpha": float, "window_size": int, "lamb": float}
	:param do_lr_atten: default False
	:param lr_atten_kwargs: {"atten_end": float, "atten_power": float}
	:param ma_period: int ∈ [1, inf); number of epochs used for averaging loss in plots and periodic reports during training.
	:param save: If True, saves model at end of training. Default False.
	:param save_path: directory in ./pretrained_models to save model
	:param update_freq: frequency of printed updates
	"""
	if do_grokfast:
		assert grokfast_kwargs is not None, "Must provide grokfast kwargs to implement grokfast"
	if do_lr_atten:
		assert lr_atten_kwargs is not None, "Must provide learn rate attenuation kwargs to implement learn rate attenuation"

	if type(model) is str:
		# Load pretrained model. Assertion will fail if model is not a child directory of ./pretrained_models.
		model, losses, test_losses, sma_losses, sma_test_losses, lrs = load_pretrained_model(model)
	else:
		losses: list[float] = list()
		test_losses: list[float] = list()
		sma_losses: list[float] = list()
		sma_test_losses: list[float] = list()
		lrs: list[float] = list()

	sma_loss = 0
	sma_test_loss = 0

	loss_func = loss_functional()
	optimizer = optimizer_func(model.parameters(), lr=lr)

	lr_init = lr

	n_train = len(train_loader)
	n_test = len(test_loader)

	long_ma_period = 4 * ma_period

	for epoch in tqdm(range(epochs)):

		model.zero_grad()
		epoch_loss = 0
		epoch_test_loss = 0
		# for grokfast
		grads = None
		# initialize training
		model.train(True)
		for batch_train_in, batch_train_out in train_loader:

			model_train_out, (_, __) = model.forward(batch_train_in)
			loss: Tensor = loss_func(model_train_out, batch_train_out.unsqueeze(-1))
			loss.backward()

			if do_grokfast:
				grads = run_grokfast(grads, model, **grokfast_kwargs)
				# grokfast often causes gradient explosion, so we want to abort if this occurs
				assert type(loss) != torch.nan, "Gradient explosion detected; loss = nan."

			epoch_loss += loss.item()
			optimizer.step()

		# prevent backprop of test error
		model.train(False)
		for batch_test_in, batch_test_out in test_loader:

			model_test_out, (_, __) = model.forward(batch_test_in)
			test_loss: Tensor = loss_func(model_test_out, batch_test_out.unsqueeze(-1))
			epoch_test_loss += test_loss.item()

		epoch_loss /= n_train
		epoch_test_loss /= n_test

		losses.append(epoch_loss)
		sma_loss = sma(losses, ma_period)
		sma_loss_long = sma(losses, long_ma_period)
		sma_losses.append(sma_loss)

		test_losses.append(epoch_test_loss)
		sma_test_loss = sma(test_losses, ma_period)
		sma_test_loss_long = sma(test_losses, long_ma_period)
		sma_test_losses.append(sma_test_loss)

		# print updates every epoch_freq epochs
		if (epoch + 1) % update_freq == 0:
			print(
				f"\nEpoch {epoch + 1} | lr = {np.format_float_scientific(lr, precision=3, min_digits=3)}\n"
				f"train loss = {np.format_float_scientific(epoch_loss, precision=3, min_digits=3)} "
				f"(MA{ma_period} = {np.format_float_scientific(sma_loss, precision=3, min_digits=3)} "
				f"| MA{long_ma_period} = {np.format_float_scientific(sma_loss_long, precision=3, min_digits=3)})\n"  
				#f" | mean err = {round(loss_to_err(epoch_loss), 3)}\n"
				f"test loss = {np.format_float_scientific(epoch_test_loss, precision=3, min_digits=3)} "
				f"(MA{ma_period} = {np.format_float_scientific(sma_test_loss, precision=3, min_digits=3)} "
				f"| MA{long_ma_period} = {np.format_float_scientific(sma_test_loss_long, precision=3, min_digits=3)}) "
				# f"| mean err = {np.format_float_scientific(loss_to_err(epoch_test_loss), precision=3)}\n"
			)

		if do_lr_atten:
			epoch_fraction = epoch / max(epochs - 1, 1)
			lr = run_lr_atten(lr_init, epoch_fraction, **lr_atten_kwargs)
			optimizer = optimizer_func(model.parameters(), lr=lr)

		lrs.append(lr)

	grokfast_function = grokfast_kwargs["func"]
	grokfast_alpha = grokfast_kwargs["alpha"]
	grokfast_window_size = grokfast_kwargs["window_size"]
	grokfast_lambda = grokfast_kwargs["lamb"]

	lr_atten_end = lr_atten_kwargs["atten_end"]
	lr_atten_pow = lr_atten_kwargs["atten_power"]

	training_log: dict[str: Any] = {
		"parameters": model.count_params(),
		"hidden size": model.hidden_size,
		"layers": model.num_layers,
		"dropout": model.lstm.dropout,
		"epochs": epochs,
		"learn rate": lr,
		"loss functional": loss_functional.__name__,
		"ma period": ma_period,
		"final train loss": losses[-1],
		"final sma train loss": sma_loss,
		"final test loss": test_losses[-1],
		"final sma test loss": sma_test_loss,
		"grokfast active": do_grokfast,
		"grokfast function": grokfast_function,
		"grokfast alpha": grokfast_alpha,
		"grokfast window size": grokfast_window_size,
		"grokfast lambda": grokfast_lambda,
		"lr atten active": do_lr_atten,
		"lr atten endpoint": lr_atten_end,
		"lr atten power": lr_atten_pow,
		"notes": ""
	}
	log_training_summary(training_log)

	if save:
		save_model(model, training_log, losses, test_losses, sma_losses, sma_test_losses, lrs, save_path)

	return


def save_model(
		model: ChemLSTM,
		training_desc_dict: dict[str: Any],
		train_losses: list[float],
		test_losses: list[float],
		sma_train_losses: list[float],
		sma_test_losses: list[float],
		learn_rates: list[float],
		model_dir: Optional[str] = None,
) -> None:
	"""
	Makes a savestate of the model after a training cycle, recording a plot of training progress both over the cycle and
	in the context of previous training cycles
	# todo remove sma train and test losses
	:param model: trained model
	:param training_desc_dict: parameters for the csv records file
	:param train_losses: all epoch train losses
	:param test_losses: all epoch test losses
	:param sma_train_losses: all epoch sma train losses
	:param sma_test_losses: all epoch sma test losses
	:param learn_rates: chronological record of learn rates by epoch over all training cycles
	:param model_dir: model name to be used as the directory for the model
	:return:
	"""
	# fallback naming convention
	if model_dir is None:
		model_dir = Path(f"./pretrained_models/{model.hidden_size}x{model.num_layers}_{datetime.strptime(datetime.now(), '%m-%d-%Y_%H-%M-%S')}")
	else:
		model_dir = Path(f"./pretrained_models/{model_dir}")

	figs_dir = model_dir / Path("figs")
	if not Path.exists(model_dir):
		# make requisite directories for storing the model and training information
		Path.mkdir(model_dir)
		Path.mkdir(figs_dir)
		is_first_save = True
	else:
		is_first_save = False

	# filenames for model descriptors and records
	model_path = model_dir / Path("model")
	log_path = model_dir / Path("records.log")
	train_loss_path = model_dir / Path("train_loss")
	test_loss_path = model_dir / Path("test_loss")
	sma_train_loss_path = model_dir / Path("sma_train_loss")
	sma_test_loss_path = model_dir / Path("sma_test_loss")
	lr_record_path = model_dir / Path("learn_rates")

	# save model
	torch.save(model.state_dict(), model_path)
	if is_first_save:
		# save model hyperparameters if not yet saved
		save_model_hyperparameters(model, log_path)
		# because this is the first save, we know the current training cycle is 0
		current_training_cycle = 0
	else:
		# otherwise, read the current log file to get the appropriate index of this training cycle
		_, current_training_cycle = parse_model_log(log_path)

	# save training cycle descriptors
	save_training_cycle(log_path, training_desc_dict, current_training_cycle)
	# do plots
	save_training_cycle_plots(
		training_desc_dict,
		current_training_cycle,
		train_losses,
		test_losses,
		sma_train_losses,
		sma_test_losses,
		learn_rates,
		figs_dir
	)

	# save serialized loss records
	# overwrite rather than appending, because we load the entire record during training
	data = (train_losses, test_losses, sma_train_losses, sma_test_losses, learn_rates)
	for idx, path in enumerate((train_loss_path, test_loss_path, sma_train_loss_path, sma_test_loss_path, lr_record_path)):
		with open(path, "wb") as file:
			pickle.dump(data[idx], file)

	print(f"Saved model to {str(model_dir)}.")

	return


def save_training_cycle_plots(
		training_dict: dict[str: Any],
		curr_cycle: int,
		train_losses: list[float],
		test_losses: list[float],
		sma_train_losses: list[float],
		sma_test_losses: list[float],
		learn_rates: list[float],
		figs_dir: Path,
) -> None:
	"""
	Saves plot for a given training cycle and updates the cumulative training plot
	Figures are of the form:
		[top]     plots of train loss, test loss, sma train loss, and sma test loss, overlayed with the loss at the same epoch
		[bottom]  plot of (sma train loss - sma test loss) to check for overfitting
	:param training_dict: csv descriptive information
	:param curr_cycle: current training cycle, starting at 0
	:param train_losses: train loss records
	:param test_losses: test loss records
	:param sma_train_losses: sma train loss records
	:param sma_test_losses: sma test loss records
	:param learn_rates: learn rate records
	:param figs_dir: directory for this model's figures
	:return: None
	"""
	# extract information from summary
	this_cycle_epochs = training_dict["epochs"]
	all_cycle_epochs = len(train_losses)
	ma_period = training_dict["ma period"]
	# get epoch indices
	# this training cycle epochs are enumerated by their position in the cumulative model training
	this_x_axis = list(range(all_cycle_epochs - this_cycle_epochs, all_cycle_epochs))
	# all epochs start at 0
	all_x_axis = list(range(all_cycle_epochs))

	# plot total improvement since model creation; overwrite the last iteration if it exists
	# plot in log values to capture scale
	all_log_train = [log(l) for l in train_losses]
	all_log_test = [log(l) for l in test_losses]
	all_log_train_sma = [log(l) for l in sma_train_losses]
	all_log_test_sma = [log(l) for l in sma_test_losses]
	all_log_lrs = [log(l) for l in learn_rates]

	this_log_train = all_log_train[-this_cycle_epochs:]
	this_log_test = all_log_test[-this_cycle_epochs:]
	this_log_train_sma = all_log_train_sma[-this_cycle_epochs:]
	this_log_test_sma = all_log_test_sma[-this_cycle_epochs:]
	this_log_lrs = all_log_lrs[-this_cycle_epochs:]

	all_log_test_train_diff = [all_log_train[idx] - all_log_test[idx] for idx in range(all_cycle_epochs)]
	all_log_test_sma_train_diff = [all_log_train_sma[idx] - all_log_test_sma[idx] for idx in range(all_cycle_epochs)]
	this_log_test_train_diff = [this_log_train[idx] - this_log_test[idx] for idx in range(this_cycle_epochs)]
	this_log_test_sma_train_diff = [this_log_train_sma[idx] - this_log_test_sma[idx] for idx in range(this_cycle_epochs)]

	# do plotting for current cycle
	fig, [ax00, ax01] = plt.subplots(2, 1, **{"figsize": (9, 9)})

	ax00.plot(this_x_axis, this_log_train, label="Train losses", color=(1, 1, 1, 0.4))  # mid grey
	ax00.plot(this_x_axis, this_log_test, label="Test losses", color=(1, 1, 1, 0.2))  # dark grey
	ax00.plot(this_x_axis, this_log_train_sma, label=f"Train loss SMA{ma_period}", color=(1, 0, 0, 0.7))
	ax00.plot(this_x_axis, this_log_test_sma, label=f"Test loss SMA{ma_period}", color=(0, 0, 1, 0.7))
	ax00.legend()
	ax00.set_title("Loss evolution graph")
	ax00.set_xlabel("Epoch")
	ax00.set_ylabel("Log loss")

	ax00a = ax00.twinx()
	ax00a.plot(this_x_axis, this_log_lrs, label="Learn rate", color=(1, 0, 1, 0.6), linestyle="--")
	ax00a.set_ylabel("Log learn rate")
	low_bound, hi_bound = min(this_log_lrs), max(this_log_lrs)
	if low_bound != hi_bound:
		ax00a.set_ylim(low_bound, hi_bound)
	ax00a.legend(loc="upper left")

	ax01.plot(this_x_axis, this_log_test_train_diff, label="train - test")
	ax01.plot(this_x_axis, this_log_test_sma_train_diff, label=f"SMA{ma_period}(train - test)")
	ax01.axhline(0, color=(1, 1, 1), linestyle="--")
	ax01.set_title("Loss difference")
	ax01.set_xlabel("Epoch")
	ax01.set_ylabel("Log loss")
	ax01.legend()

	# save new plot for loss over just this training cycle
	fig.suptitle(f"Cycle {curr_cycle}")
	this_cycle_path = figs_dir / Path(f"cycle_{curr_cycle}.png")
	plt.savefig(this_cycle_path)

	# do plotting for all cycles
	fig, [ax00, ax01] = plt.subplots(2, 1, **{"figsize": (9, 9)})

	ax00.plot(all_x_axis, all_log_train, label="Train losses", color=(1, 1, 1, 0.4))  # mid grey
	ax00.plot(all_x_axis, all_log_test, label="Test losses", color=(1, 1, 1, 0.2))  # dark grey
	# drop sma period from the label here because it may differ between cycles
	ax00.plot(all_x_axis, all_log_train_sma, label="Train loss SMA", color=(1, 0, 0, 0.7))
	ax00.plot(all_x_axis, all_log_test_sma, label="Test loss SMA", color=(0, 0, 1, 0.7))
	ax00.legend()
	ax00.set_title("Loss evolution graph")
	ax00.set_xlabel("Epoch")
	ax00.set_ylabel("Log loss")
	# todo do axvlines to separate cycles
	# need to parse training cycle lengths via model log file

	ax00a = ax00.twinx()
	ax00a.plot(all_x_axis, all_log_lrs, label="Learn rate", color=(1, 0, 1, 0.6), linestyle="--")
	ax00a.set_ylabel("Log learn rate")
	low_bound, hi_bound = min(all_log_lrs), max(all_log_lrs)
	if low_bound != hi_bound:
		ax00a.set_ylim(low_bound, hi_bound)
	ax00a.legend(loc="upper left")

	ax01.plot(all_x_axis, all_log_test_train_diff, label="train - test")
	ax01.plot(all_x_axis, all_log_test_sma_train_diff, label="SMA(train - test)")
	ax01.axhline(0, color=(1, 1, 1), linestyle="--")
	ax01.set_title("Loss difference")
	ax01.set_xlabel("Epoch")
	ax01.set_ylabel("Log loss")
	ax01.legend()

	# overwrite plot for loss over all training cycles
	fig.suptitle(f"All cycles")
	all_cycle_path = figs_dir / Path(f"all_training_cycles.png")
	plt.savefig(all_cycle_path)

	return None


def save_model_hyperparameters(model: ChemLSTM, log_path: Path) -> None:
	"""
	Saves model hyperparameter information
	:param model: model to save
	:param log_path: path to this model's log
	:return: None
	"""

	input_size = model.input_size
	output_size = model.output_size
	hidden_size = model.hidden_size
	layers = model.num_layers
	dropout = model.dropout

	attribute_names = ("input_size", "output_size", "hidden_size", "num_layers", "dropout")

	with open(log_path, "w") as log_file:
		log_file.write(f"#model_parameters\n")
		for idx, attribute in enumerate((input_size, output_size, hidden_size, layers, dropout)):
			# use leading "%" to denote parameters
			log_file.write(f"%{attribute_names[idx]}={attribute}\n")

	return


def save_training_cycle(
		log_path: Path,
		training_desc_dict: dict[str: any],
		current_cycle: int,
) -> None:
	"""
	Append information in training cycle summary to model log
	:param log_path: model log path
	:param training_desc_dict: training cycle summary information
	:param current_cycle: index of model's current training cycle
	:return:
	"""

	with open(log_path, "a") as log_file:
		log_file.write(f"#train {current_cycle}\n")
		for k, v in training_desc_dict.items():
			log_file.write(f"%{k}={v}\n")

	return


def parse_model_log(log_path: Path) -> tuple[dict[str: int|float], int]:
	"""
	Parses model logfile
	:param log_path: Path of model log
	:return: list of [model.input_size, model.output_size, model.hidden_size, model.num_layers, model.dropout] and the
		index of the current training cycle
	"""

	with open(log_path, "r") as desc_file:
		lines: list[str] = list(desc_file.readlines())

	separators = [idx for idx, line in enumerate(lines) if line[0] == "#"]
	model_param_start, model_param_end = separators[0], separators[1]
	# use [1:-1] to drop the "%" parameter delimiter and newline
	# split on "=" to get key value pairs and keep only the value
	attributes = [line[1:-1].split("=")[1] for line in lines[model_param_start + 1: model_param_end]]
	attributes_parsed = [
		int(attributes[0]),  # input_size
		int(attributes[1]),  # output_size
		int(attributes[2]),  # hidden_size
		int(attributes[3]),  # num_layers
		float(attributes[4]),  # dropout
	]
	completed_training_cycles: int = len(separators) - 1
	attributes_dict: dict[str: int|float] = {
		"input_size": attributes_parsed[0],
		"output_size": attributes_parsed[1],
		"hidden_size": attributes_parsed[2],
		"num_layers": attributes_parsed[3],
		"dropout": attributes_parsed[4],
	}

	return attributes_dict, completed_training_cycles


def load_pretrained_model(model_folder: str,
		) -> tuple[ChemLSTM, list[float], list[float], list[float], list[float], list[float]]:
	"""
	Loads pretrained model and chronological training record
	:param model_folder: location of model files
	:return: tuple of (model, train losses, test losses, sma train losses, sma test losses, learn rates)
	"""
	model_dir = Path(f"./pretrained_models/{model_folder}")
	assert Path.exists(model_dir), f"{model_dir} does not exist."

	device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

	model_path = model_dir / Path("model")
	log_path = model_dir / Path("records.log")
	train_loss_path = model_dir / Path("train_loss")
	test_loss_path = model_dir / Path("test_loss")
	sma_train_loss_path = model_dir / Path("sma_train_loss")
	sma_test_loss_path = model_dir / Path("sma_test_loss")
	lr_record_path = model_dir / Path("learn_rates")

	hyperparameter_dict, last_cycle = parse_model_log(log_path)
	(input_size, output_size, hidden_size, layers, dropout) = hyperparameter_dict.values()
	model = ChemLSTM(
		input_size=input_size,
		output_size=output_size,
		hidden_size=hidden_size,
		num_layers=layers,
		dropout=dropout,
		device=device
	)
	model.load_state_dict(torch.load(model_path))

	with open(train_loss_path, "rb") as train_loss_file:
		train_losses: list[float] = pickle.load(train_loss_file)
	with open(test_loss_path, "rb") as test_loss_file:
		test_losses: list[float] = pickle.load(test_loss_file)
	with open(sma_train_loss_path, "rb") as sma_train_loss_file:
		sma_train_losses: list[float] = pickle.load(sma_train_loss_file)
	with open(sma_test_loss_path, "rb") as sma_test_loss_file:
		sma_test_losses: list[float] = pickle.load(sma_test_loss_file)
	with open(lr_record_path, "rb") as lr_record_file:
		learn_rates: list[float] = pickle.load(lr_record_file)

	return model, train_losses, test_losses, sma_train_losses, sma_test_losses, learn_rates


def log_training_summary(training_log: dict[str: Any]):
	"""
	Records a log of this model's training cycle for comparison against other models
	:param training_log: model training logfile
	:return:
	"""
	log_path = "./logs/log.csv"
	try:
		current_train_logs = pd.read_csv(log_path, header=0, index_col=0)
		n_logs = current_train_logs.shape[0]
		current_train_logs = pd.concat((current_train_logs, pd.DataFrame(training_log, index=[n_logs])), axis="rows")
	except Exception as e:
		print(e)
		current_train_logs = pd.DataFrame(training_log, index=[0])

	current_train_logs.to_csv(log_path)

	return


def get_dataloaders(data_in: Tensor, data_out: Tensor, batch_size: int = 32, train_pct: float = 0.85) -> tuple[DataLoader, DataLoader]:
	"""
	Retrieves DataLoaders for training
	:param data_in: Input data
	:param data_out: Output data
	:param batch_size: Batch size
	:param train_pct: Percent of data used in training
	:return: training dataloader, testing dataloader
	"""
	train_cutoff = int(data_in.shape[0] * train_pct)

	train_data_in = data_in[:train_cutoff]
	train_data_out = data_out[:train_cutoff]
	test_data_in = data_in[train_cutoff:]
	test_data_out = data_out[train_cutoff:]

	train_dataset = TensorDataset(train_data_in, train_data_out)
	test_dataset = TensorDataset(test_data_in, test_data_out)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

	return train_loader, test_loader


def standard_scale(this_data: Tensor) -> tuple[Tensor, tuple[float, float]]:
	"""
	Scales this_data to have standard deviation of 1 and mean of 0
	"""

	data_mean = torch.mean(this_data)
	data_std = torch.std(this_data)

	data_out = (this_data - data_mean) / data_std
	return data_out, (data_mean.item(), data_std.item())


def inverse_standard_scale(scaled_data: Tensor, data_mean: float, data_std: float) -> Tensor:
	"""
	Inverses standard_scale() given the pre-scaled mean and standard deviation of scaled_data
	"""
	descaled_data = scaled_data * data_std + data_mean
	return descaled_data


def minmax_scale(data_: Tensor, min_: float, max_: float) -> tuple[Tensor, Tensor, Tensor]:
	"""
	Scales data, preserving relative differences, to have a specified range (min_, max_)
	:param data_: data to scale
	:param min_: new data minimum
	:param max_: new data maximum
	:return: scaled data and a tuple of the prescaled min and max values for inverse scaling
		*note: to inverse standard scale, feed the output back into this function
	"""
	data_min = torch.min(data_)
	data_max = torch.max(data_)
	data_range = data_max - data_min
	out_range = max_ - min_

	# set data minimum to 0
	data_scaled = data_ - data_min
	# set data to range [0, 1]
	data_scaled = data_scaled / data_range
	# scale to desired scale and shift
	data_scaled = data_scaled * out_range + min_
	# print(f"scaling min {data_min} to {min_} | scaling max {data_max} to {max_} | scale factor: {data_range / out_range}")

	return data_scaled, data_min, data_max


def drop_above_diagonal(matrix: Tensor) -> Tensor:
	"""
	Filters matrix to only values on or below diagonal row-wise, and concatenates resultant vectors
	:param matrix: square matrix
	:return: Matrix values on or below diagonal concatenated to a single dimension
	"""
	sh = matrix.shape
	assert sh[0] == sh[1], f"Expected square matrix, got side lengths {sh}."
	side: int = sh[0]
	vecs_out = [matrix[s, -s:] for s in range(side)]
	vec_out = torch.cat(vecs_out, dim=-1)

	return vec_out


def load_training_data() -> tuple[Tensor, Tensor, Tensor, Tensor, list[str]]:
	"""
	Loads training data Tensors
	:return: bond order matrix, connection matrix, distance matrix, log kow, and molecule names for each molecule in
		the experiment
	"""
	bxpath = "./training_data/bond_order_matrix_randomized"
	cxpath = "./training_data/connection_matrix_randomized"
	dxpath = "./training_data/distance_matrix_randomized"
	kowpath = "./training_data/kow_vector_randomized"

	orders = torch.load(bxpath)
	connection = torch.load(cxpath)
	distance = torch.load(dxpath)
	log_kow = torch.load(kowpath)
	with open("./training_data/data_ids.txt") as ids:
		content = ids.read()
		filenames = content.split("\n")

	return orders, connection, distance, log_kow, filenames


def loss_atten_destabilize(loss_ledger: list[float], last_lr: float, ma: float, ma_period: int):

	"""
	WIP; too unstable
	:param loss_ledger:
	:param last_lr:
	:param ma:
	:param ma_period:
	:return:
	"""
	ma_period = min(ma_period, len(loss_ledger))
	min_loss = min(loss_ledger)
	curr_loss = loss_ledger[-1]
	best_diff = curr_loss - min_loss
	loss_ledger_tensor = Tensor(loss_ledger[-ma_period:])
	ma_diff = ma - curr_loss
	ma_direction = -1 if ma_diff > 0 else 1
	ma_movement = torch.std(loss_ledger_tensor) / torch.mean(loss_ledger_tensor)
	# if sma is significantly above current loss or we are currently at a loss minimum, decrease lr:
	lr_mod = min(max(-1.0, ma_direction * best_diff / ma_movement), 1.0) + (torch.rand(1) * last_lr / 10).item()
	print(f"{lr_mod = }")

	return max(min(last_lr * (1 + lr_mod / 2), 0.5), 1e-5)


def lr_atten_parabolic(start: float, end: float, epoch_fraction: float, c: float = 15.0):
	"""
	Modifies loss by the formula:
		next = l0 * (1 - x^(1/c)) + ln * x;
		l0 = start;
		ln = end;
		x = epoch_fraction;
		c = c.
		# todo save the desmos graph and incl a link here
	:param start: initial lr
	:param end: final lr
	:param epoch_fraction: fraction of training cycle complete
	:param c: constant, default 15.0, which introduces very agressive attenuation to start with stable intermediates
	:return:
	"""
	next_lr = start * (1 - epoch_fraction**(1 / c)) + epoch_fraction * end
	return next_lr


def lr_atten_ma_crossover(last_lr: float, short_ma_loss: float, long_ma_loss: float) -> float:
	"""
	WIP; oscillates
	:param last_lr:
	:param short_ma_loss:
	:param long_ma_loss:
	:return:
	"""

	max_change = 1.01
	# decrements lr if short ma is below long ma (currently improving); increments otherwise
	change = short_ma_loss / long_ma_loss
	change = min(max(change, 1 / max_change), max_change)  # clip to [95%, 105%]
	lr_out = min(last_lr * change, 0.001)

	return lr_out


def get_training_data_tensors() -> tuple[Tensor, Tensor]:
	# retrieve data, stacked along 0th dimension
	bx, cx, dx, ex, names = load_training_data()
	# cx, (cx_min, cx_max) = minmax_scale(cx, -1, 1)

	# check of gpu is accessible and use it if possible, else do computation on cpu
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
	print(f"{device = }")
	# move training data to this device
	bx = bx.to(device)
	cx = cx.to(device)
	# dx = dx.to(device)  # ignore dx because it is not used in training
	ex = ex.to(device)

	# normalize data for training
	cx = torch.tanh(cx / 12)  # gives C-C bonds near 0.5; X bonds near 1, and H bonds near 0.1
	ex_norm, e_min, e_max = minmax_scale(ex, 0, 1)

	# extract number of molecules in the training dataset
	bxcx = torch.concat((bx, cx), dim=-1)
	return bxcx, ex_norm


def expt_0():

	# initialize model
	in_size, out_size = 60, 1
	hidden_size, num_layers, dropout = 128, 2, 0.1
	model = ChemLSTM(in_size, hidden_size, out_size, num_layers, dropout, )

	# constants
	save_out = True
	bs = 2**9  # most efficient batch size found
	bxcx, ex = get_training_data_tensors()
	train_loader, test_loader = get_dataloaders(bxcx, ex, bs, 0.85)
	lf = L1Loss
	gfa, gfw, gfl, gff = 0.999, 30, 1.0, "ema"

	# train 0 - no grokfast and lr atten from 1e-3 to 1e-5
	"""lr_0 = 0.001
	es = 2**9
	do_gf = False
	do_atten = True
	gf_kwargs = {"func": gff, "alpha": gfa, "window_size": gfw, "lamb": gfl}
	lr_atten_kwargs = {"atten_end": 1e-5, "atten_power": 15.0}
	train_model_verbose(
		model, train_loader, test_loader, es, loss_functional=lf, lr=lr_0,
		do_grokfast=do_gf, grokfast_kwargs=gf_kwargs, do_lr_atten=do_atten, lr_atten_kwargs=lr_atten_kwargs,
		save_path="expt_0", save=save_out,
	)

	# train 1 - grokfast ema and lr atten from 1e-5 to 1e-7
	lr_0 = 1e-5
	es = 2 ** 12
	do_gf = True
	do_atten = True
	gf_kwargs = {"func": gff, "alpha": gfa, "window_size": gfw, "lamb": gfl}
	lr_atten_kwargs = {"atten_end": 1e-7, "atten_power": 15.0}
	train_model_verbose(
		"expt_0", train_loader, test_loader, es, loss_functional=lf, lr=lr_0,
		do_grokfast=do_gf, grokfast_kwargs=gf_kwargs, do_lr_atten=do_atten, lr_atten_kwargs=lr_atten_kwargs,
		save_path="expt_0", save=save_out,
	)"""
	# progress extremely slow; final train = 0.183, test 0.158

	# train 2 -
	# no lr atten (constant at 5e-5)
	# increase gradient accumulation by changing grokfast function to ma (from ema) and lambda to 2.0 (from 1.0)
	lr_0 = 5e-5
	es = 2 ** 12
	do_gf = True
	do_atten = False
	gf_kwargs = {"func": "ma", "alpha": 0.98, "window_size": 64, "lamb": 2.0}
	lr_atten_kwargs = {"atten_end": 1e-6, "atten_power": 15.0}
	train_model_verbose(
		"expt_0", train_loader, test_loader, es, loss_functional=lf, lr=lr_0,
		do_grokfast=do_gf, grokfast_kwargs=gf_kwargs, do_lr_atten=do_atten, lr_atten_kwargs=lr_atten_kwargs,
		save_path="expt_0", save=save_out,
	)


def expt_1():
	name = "expt_1"

	# initialize model
	in_size, out_size = 60, 1
	hidden_size, num_layers, dropout = 128, 2, 0.1
	model = ChemLSTM(in_size, hidden_size, out_size, num_layers, dropout, )

	# constants
	save_out = True
	bs = 2**9  # most efficient batch size found
	bxcx, ex = get_training_data_tensors()
	# we can't generate new dataloaders at any point as this would cause data leakage
	train_loader, test_loader = get_dataloaders(bxcx, ex, bs, 0.85)
	lf = L1Loss
	gfa, gfw, gfl, gff = 0.999, 30, 1.0, "ema"

	"""# train 0 - no grokfast and lr atten from 1e-3 to 1e-5
	lr_0 = 0.001
	es = 2**6
	do_gf = False
	do_atten = True
	gf_kwargs = {"func": gff, "alpha": gfa, "window_size": gfw, "lamb": gfl}
	lr_atten_kwargs = {"atten_end": 1e-4, "atten_power": 15.0}
	train_model_verbose(
		model, train_loader, test_loader, es, loss_functional=lf, lr=lr_0,
		do_grokfast=do_gf, grokfast_kwargs=gf_kwargs, do_lr_atten=do_atten, lr_atten_kwargs=lr_atten_kwargs,
		save_path=name, save=save_out,
	)

	# train 1 - grokfast ma, no lr atten
	lr_0 = 1e-4
	es = 2 ** 8
	do_gf = True
	do_atten = False
	gf_kwargs = {"func": "ma", "alpha": 0.98, "window_size": 64, "lamb": 2.0}
	lr_atten_kwargs = {"atten_end": 1e-7, "atten_power": 15.0}
	train_model_verbose(
		name, train_loader, test_loader, es, loss_functional=lf, lr=lr_0,
		do_grokfast=do_gf, grokfast_kwargs=gf_kwargs, do_lr_atten=do_atten, lr_atten_kwargs=lr_atten_kwargs,
		save_path=name, save=save_out,
	)
	# progress extremely slow; final train = 0.183, test 0.158

	# train 2 -
	# same grokfast settings; lr atten from 1e-4 to 2e-5, less aggresive atten (pow=10 from 15)
	lr_0 = 1e-4
	es = 2 ** 10
	do_gf = True
	do_atten = False
	gf_kwargs = {"func": "ma", "alpha": 0.98, "window_size": 64, "lamb": 2.0}
	lr_atten_kwargs = {"atten_end": 2e-5, "atten_power": 10.0}
	train_model_verbose(
		name, train_loader, test_loader, es, loss_functional=lf, lr=lr_0,
		do_grokfast=do_gf, grokfast_kwargs=gf_kwargs, do_lr_atten=do_atten, lr_atten_kwargs=lr_atten_kwargs,
		save_path=name, save=save_out,
	)"""
	# train 3
	# linear lr decay (c = 1) with same boundaries as last cycle
	# more agressive grokfast (lamb=2 -> lamb=5)
	# 1024 epochs -> 4096 epochs
	lr_0 = 1e-4
	es = 2 ** 12
	do_gf = True
	do_atten = True
	gf_kwargs = {"func": "ma", "alpha": 0.98, "window_size": 64, "lamb": 5.0}
	lr_atten_kwargs = {"atten_end": 5e-5, "atten_power": 1.0}
	train_model_verbose(
		name, train_loader, test_loader, es, loss_functional=lf, lr=lr_0,
		do_grokfast=do_gf, grokfast_kwargs=gf_kwargs, do_lr_atten=do_atten, lr_atten_kwargs=lr_atten_kwargs,
		save_path=name, save=save_out,
	)

	return


def expt_2():
	name = "expt_2"

	# initialize model
	in_size, out_size = 60, 1
	hidden_size, num_layers, dropout = 512, 2, 0.1
	model = ChemLSTM(in_size, hidden_size, out_size, num_layers, dropout, )

	# constants
	save_out = True
	bs = 2**9  # most efficient batch size found
	bxcx, ex = get_training_data_tensors()
	# we can't generate new dataloaders at any point as this would cause data leakage
	train_loader, test_loader = get_dataloaders(bxcx, ex, bs, 0.85)
	lf = L1Loss
	gfa, gfw, gfl, gff = 0.999, 30, 1.0, "ema"

	"""# train 0 -
	# no grokfast and no atten, 128 epochs @ 0.001 lr
	lr_0 = 0.001
	es = 2**7
	do_gf = False
	do_atten = False
	gf_kwargs = {"func": gff, "alpha": gfa, "window_size": gfw, "lamb": gfl}
	lr_atten_kwargs = {"atten_end": 1e-4, "atten_power": 15.0}
	train_model_verbose(
		model, train_loader, test_loader, es, loss_functional=lf, lr=lr_0,
		do_grokfast=do_gf, grokfast_kwargs=gf_kwargs, do_lr_atten=do_atten, lr_atten_kwargs=lr_atten_kwargs,
		save_path=name, save=save_out,
	)

	# train 1 -
	# lr atten decreasing linearly from 0.001 to 0.0001 over 256 epochs
	lr_0 = 0.001
	es = 2 ** 8
	do_gf = False
	do_atten = True
	gf_kwargs = {"func": "ma", "alpha": 0.98, "window_size": 64, "lamb": 2.0}
	lr_atten_kwargs = {"atten_end": 0.0001, "atten_power": 1.0}
	train_model_verbose(
		name, train_loader, test_loader, es, loss_functional=lf, lr=lr_0,
		do_grokfast=do_gf, grokfast_kwargs=gf_kwargs, do_lr_atten=do_atten, lr_atten_kwargs=lr_atten_kwargs,
		save_path=name, save=save_out,
	)

	# train 2 -
	# grokfast ema, with more agressive atten (c=5) from 1e-4 to 2e-5, over 1024 epochs
	lr_0 = 1e-4
	es = 2 ** 10
	do_gf = True
	do_atten = True
	gf_kwargs = {"func": "ema", "alpha": 0.98, "window_size": 64, "lamb": 2.0}
	lr_atten_kwargs = {"atten_end": 2e-5, "atten_power": 5.0}
	train_model_verbose(
		name, train_loader, test_loader, es, loss_functional=lf, lr=lr_0,
		do_grokfast=do_gf, grokfast_kwargs=gf_kwargs, do_lr_atten=do_atten, lr_atten_kwargs=lr_atten_kwargs,
		save_path=name, save=save_out,
	)"""

	"""
	# note - this did not work, lr too low, appears to be at a local min
	# train 3
	# decrease to a final lr of 1e-6 over 8192 cycles, with slightly more agressive ema (lamb: 2.0 -> 2.5) and atten (pow: 5 -> 6)
	lr_0 = 2e-5
	es = 2 ** 13
	do_gf = True
	do_atten = True
	gf_kwargs = {"func": "ema", "alpha": 0.98, "window_size": 64, "lamb": 2.5}
	lr_atten_kwargs = {"atten_end": 1e-6, "atten_power": 6.0}
	train_model_verbose(
		name, train_loader, test_loader, es, loss_functional=lf, lr=lr_0,
		do_grokfast=do_gf, grokfast_kwargs=gf_kwargs, do_lr_atten=do_atten, lr_atten_kwargs=lr_atten_kwargs,
		save_path=name, save=save_out,
	)"""

	# train 3
	# inc lr 7.5x to 1.5e-4; use aggresive grokfast ma (64 lookbacks, lamb=2.5)
	lr_0 = 1.5e-4
	es = 2 ** 13
	do_gf = True
	do_atten = False
	gf_kwargs = {"func": "ma", "alpha": 0.95, "window_size": 64, "lamb": 5.0}
	lr_atten_kwargs = {"atten_end": 1e-6, "atten_power": 6.0}
	train_model_verbose(
		name, train_loader, test_loader, es, loss_functional=lf, lr=lr_0,
		do_grokfast=do_gf, grokfast_kwargs=gf_kwargs, do_lr_atten=do_atten, lr_atten_kwargs=lr_atten_kwargs,
		save_path=name, save=save_out,
	)

	# train 4
	# all parameters same as train 3 but with longer gf window size (64 -> 128)
	lr_0 = 1.5e-4
	es = 2 ** 13
	do_gf = True
	do_atten = False
	gf_kwargs = {"func": "ma", "alpha": 0.95, "window_size": 64, "lamb": 5.0}
	lr_atten_kwargs = {"atten_end": 1e-6, "atten_power": 6.0}
	train_model_verbose(
		name, train_loader, test_loader, es, loss_functional=lf, lr=lr_0,
		do_grokfast=do_gf, grokfast_kwargs=gf_kwargs, do_lr_atten=do_atten, lr_atten_kwargs=lr_atten_kwargs,
		save_path=name, save=save_out,
	)

	return


if __name__ == '__main__':

	torch.manual_seed(492)
	expt_2()
