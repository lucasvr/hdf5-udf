#
# Simple example: output a dataset made of variable strings
#
# To embed it in an existing HDF5 file, run:
# $ make files
# $ hdf5-udf example-string.h5 test-string-output.py RollingStone.py:405:string(5)
#

def dynamic_dataset():

    # This is the data we want to output from the UDF. Each one of the
    # 405 strings (group of characters) becomes a variable-sized string.

    lyrics = """
    Once upon a time, you dressed so fine
    Threw the bums a dime in your prime, didn't you?
    People'd call, say: Beware, doll! You're bound to fall!
    You thought they were all kiddin' you
    You used to laugh about
    Everybody that was hangin' out
    Now you don't talk so loud
    Now you don't seem so proud
    About having to be scrounging your next meal

    How does it feel?
    How does it feel?
    To be on your own?
    With no direction home?
    Like a complete unknown?
    Like a rolling stone?

    You went to the finest school, all right, miss lonely
    But you know you only used to get juiced in it
    Nobody has ever taught you how to live out on the street
    And now you're gonna have to get used to it
    You said you'd never compromise
    With the mystery tramp, but now you realize
    He's not selling any alibis
    As you stare into the vacuum of his eyes
    And say: Do you want to make a deal?

    How does it feel?
    How does it feel?
    To be on your own?
    With no direction home?
    Like a complete unknown?
    Like a rolling stone?

    You never turned around to see the frowns on the jugglers and the clowns
    When they all did tricks for you
    You never understood that it ain't no good
    You shouldn't let other people get your kicks for you
    You used to be so amused
    At napoleon in rags and the language that he used
    Go to him now, he calls you, you can't refuse!
    When you ain't got nothing, you got nothing to lose
    You're invisible now, you got no secrets to conceal

    How does it feel?
    How does it feel?
    To be on your own?
    With no direction home?
    Like a complete unknown?
    Like a rolling stone?

    Princess on the steeple and all the pretty people
    They're all drinkin', thinkin' that they got it made
    Exchangin' all precious gifts, but I think you better take your diamond ring
    I think you better pawn it, babe!
    Used to ride on a chrome horse with your diplomat
    Who carried on his shoulder a siamese cat
    Ain't it hard when you discover that
    He really wasn't where it's at
    After he took from you everything he could steal

    How does it feel?
    How does it feel?
    To be on your own?
    With no direction home?
    Like a complete unknown?
    Like a rolling stone?
    """
    words = lyrics.split()

    udf_data = lib.getData("RollingStone.py")
    udf_dims = lib.getDims("RollingStone.py")

    for i in range(udf_dims[0]):
        if i < len(words):
            lib.setString(udf_data[i], words[i].encode("utf-8"))
