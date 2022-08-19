from pydub import AudioSegment

def trim_audio(start_min: int, start_sec: int, sound_file, f_format):

    start = start_min*60*1000+start_sec*1000
    sound = AudioSegment.from_file(file=sound_file, format=f_format)
    extract = sound[start:]
    extract.export(sound_file[:-4] + "_trim.wav", format="wav")

def change_volume(mode, sound_file, amount, f_format):
    sound = AudioSegment.from_file(file=sound_file, format=f_format)
    if mode == "inc":
        new_sound = sound + amount 
    else:
        new_sound = sound - amount 
    new_sound.export(sound_file[:-4] + "_vol.wav", format="wav")

if __name__ == "__main__":

    start_min = 0
    start_sec = 9
    sound_file = "/home2/debnaths/streamlined/data/puma/da_mauwabote.wav"
    f_format = "wav"

    trim_audio(start_min, start_sec, sound_file, f_format)