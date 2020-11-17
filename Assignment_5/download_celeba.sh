#!/usr/bin/env bash

# download 
curl -O https://uofi.box.com/shared/static/q4pf89jtkvjndi4f8ip7wofuulhhphjj.zip
mkdir celeba_data
unzip q4pf89jtkvjndi4f8ip7wofuulhhphjj.zip -d celeba_data
rm q4pf89jtkvjndi4f8ip7wofuulhhphjj.zip
