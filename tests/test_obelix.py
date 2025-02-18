def test_OBELiX():
    from obelix import OBELiX
    obelix = OBELiX("test_data")
    assert len(obelix.dataframe) == 599

if __name__ == "__main__":
    test_OBELiX()
