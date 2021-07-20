//
// Created by matteo on 7/20/21.
//
#pragma once

/*
* This class represent the device, acting as the device API
*/
class Device {
private:
public:
    /* Initialize the device with a WxH simulation */
    void init(unsigned int w, unsigned int h);
    void step();
    void blit();
};

    /* It is the actual reference to the device */



