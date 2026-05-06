#!/bin/sh
#python main_train_egocast.py -opt options/train_ec_60f_sf_r_4_on.json &
#python main_train_egocast.py -opt options/train_ec_90f_sf_r_4_on.json &
#python main_train_egocast.py -opt options/train_ec_120f_sf_r_4_on.json &
#python main_train_egocast.py -opt options/train_ec_150f_sf_r_4_on.json &
python main_train_egocast.py -opt options/train_ec_f_60.json &
python main_train_egocast.py -opt options/train_ec_f_90.json &
python main_train_egocast.py -opt options/train_ec_f_120.json &
python main_train_egocast.py -opt options/train_ec_f_150.json &
