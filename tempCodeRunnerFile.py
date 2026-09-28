    elif (
        "most units" in q
        or "most quantity" in q
        or "sold the most" in q
        or "sold most" in q
    ):

        if "Quantity" in data.columns:

            product_data = (
                data.groupby("Product")["Quantity"]
                .sum()
            )

            product_name = product_data.idxmax()
            product_quantity = product_data.max()

            return (
                f"The product with the highest number of "
                f"units sold is {product_name}, with "
                f"{product_quantity} units."
            )

        else:

            return (
                "Quantity information is not available "
                "in the dataset."
            )


    # --------------------------------------------------------
    # REGION WITH MOST REVENUE
    # --------------------------------------------------------

    elif (
        "top region" in q
        or "best region" in q
        or "highest revenue region" in q
        or "region most revenue" in q
    ):

        region_data = (
            data.groupby("Region")["Revenue"]
            .sum()
        )
