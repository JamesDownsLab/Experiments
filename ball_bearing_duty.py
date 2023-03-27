m = 0.006
c = -1.82

def duty_to_peak_to_peak(duty):
    return duty * m + c

def peak_to_peak_to_acceleration(volts):
    amplitude = volts / 2
    acceleration = amplitude * 4.54
    return acceleration

def duty_to_dimensionless_acceleration(duty):
    volts = duty_to_peak_to_peak(duty)
    accel = peak_to_peak_to_acceleration(volts)
    return accel

def dimensionless_acceleration_to_duty(acc):
    amplitude = acc / 4.54
    volts = amplitude * 2
    duty = (volts - c)/m
    return duty

if __name__ == '__main__':
    print(duty_to_dimensionless_acceleration(470
                                            ))
