def test_OBELiX():
    from obelix import OBELiX
    import shutil

    shutil.rmtree("rawdata", ignore_errors=True)
    obelix = OBELiX()
    assert len(obelix) == 599
    assert len(obelix.test_dataset) == 121
    assert len(obelix.train_dataset) == 478
    assert obelix[0]["ID"] == "jqc"
    assert obelix["jqc"]["ID"] == "jqc"

if __name__ == "__main__":
    test_OBELiX()
