#!/bin/sh

./vtm/bin/VTM10Enc -c ./vtm/cfg/encoder_lowdelay_vtm.cfg -c ./vtm/cfg/per-sequence/43.cfg -c ./vtm/cfg/formatRGB.cfg -q $2 -i $1_org.rgb -wdt $3 -hgt $4 -o $1_rec.rgb -b $1.bin >>$1.log 
