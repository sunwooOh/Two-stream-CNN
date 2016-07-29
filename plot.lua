require 'gnuplot'

function plot (x_val, y_val, x_label, y_label, title)
	fname = title:gsub("%s+", "") .. '.png'
	p = gnuplot.pngfigure (save_path .. fname)
		
	gnuplot.grid (true)
	gnuplot.title (title)
	gnuplot.xlabel (x_label)
	gnuplot.ylabel (y_label) 

	gnuplot.plot (y_val, '-+')

	gnuplot.plotflush ()
	gnuplot.close (p)
end

root_path = 'save_data/temporal/'
dir_name = {
	'lr0.001bat5ti9300_wd1e-4gc5/',
	'lr0.0001bat5ti9300/',
}
file_name = 'tables.dat'
save_path = 'cat_plots/'

for i = 1, #dir_name do
	data_table = torch.load (root_path .. dir_name[i] .. file_name)

	tr_ten = data_table.tr_acc
	vl_ten = data_table.vl_acc
	epc_ten = data_table.ten_ep
	ls_ten = data_table.tr_loss

	if i == 1 then
		tr_acc = tr_ten
		vl_acc = vl_ten
		epochs = epc_ten
		tr_loss = ls_ten
	else
		tr_acc = torch.cat (tr_acc, tr_ten, 1)
		vl_acc = torch.cat (vl_acc, vl_ten, 1)
		tr_loss = torch.cat (tr_loss, ls_ten, 1)
			
	end
end


titles = {
	'Training Accuracy',
	'Validation Accuracy',
	'Training Loss'
}

plot (nil, tr_acc, 'Epoch', 'Accuracy', 'Training Accuracy')
plot (nil, vl_acc, 'Epoch', 'Accuracy', 'Validation Accuracy')
--plot (epochs, tr_loss, 'Epoch', 'Loss', 'Training Loss')

mult_title = 'Training and Validation Accuracy'
fname = mult_title:gsub("%s+", "") .. '.png'

p = gnuplot.pngfigure (save_path .. fname)

gnuplot.grid (true)
gnuplot.title (mult_title)
gnuplot.xlabel ('Epoch')
gnuplot.ylabel ('Accuracy')

gnuplot.plot (
	{ 'Training', tr_acc },
	{ 'Validation', vl_acc }
)

gnuplot.plotflush()
gnuplot.close (p)
