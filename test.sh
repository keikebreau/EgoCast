#!/bin/sh
#python main_test_egocast_forecasting.py -opt options/test_egocast_forecasting.json --checkpoint results/results_ec_60_sf_r_4_on/EgoExo/models/30000_G.pth --slowfast-recent-frames 4 --slowfast-use-recent-delta --future-frames 60 --forecast-head slowfast --gt-input
#read -p "Press [Enter] to continue" REPLY
#python main_test_egocast_forecasting.py -opt options/test_egocast_forecasting.json --checkpoint results/results_ec_90_sf_r_4_on/EgoExo/models/30000_G.pth --slowfast-recent-frames 4 --slowfast-use-recent-delta --future-frames 90 --forecast-head slowfast --gt-input
#read -p "Press [Enter] to continue" REPLY
#python main_test_egocast_forecasting.py -opt options/test_egocast_forecasting.json --checkpoint results/results_ec_120_sf_r_4_on/EgoExo/models/30000_G.pth --slowfast-recent-frames 4 --slowfast-use-recent-delta --future-frames 120 --forecast-head slowfast --gt-input
#read -p "Press [Enter] to continue" REPLY
#python main_test_egocast_forecasting.py -opt options/test_egocast_forecasting.json --checkpoint results/results_ec_150_sf_r_4_on/EgoExo/models/30000_G.pth --slowfast-recent-frames 4 --slowfast-use-recent-delta --future-frames 150 --forecast-head slowfast --gt-input
#read -p "Press [Enter] to continue" REPLY

python main_test_egocast_forecasting.py -opt options/test_egocast_forecasting.json --checkpoint results/results_EgoCast_60Frames/EgoExo/models/30000_G.pth --future-frames 60 --gt-input
read -p "Press [Enter] to continue" REPLY
python main_test_egocast_forecasting.py -opt options/test_egocast_forecasting.json --checkpoint results/results_EgoCast_90Frames/EgoExo/models/30000_G.pth --future-frames 90 --gt-input
read -p "Press [Enter] to continue" REPLY
python main_test_egocast_forecasting.py -opt options/test_egocast_forecasting.json --checkpoint results/results_EgoCast_120Frames/EgoExo/models/30000_G.pth --future-frames 120 --gt-input
read -p "Press [Enter] to continue" REPLY
python main_test_egocast_forecasting.py -opt options/test_egocast_forecasting.json --checkpoint results/results_EgoCast_150Frames/EgoExo/models/30000_G.pth --future-frames 150 --gt-input
read -p "Press [Enter] to continue" REPLY
