import pandas as pd
import usaddress


class Preprocess:
    def __init__(self, dataframes):
        self.dataframes = dataframes

    # Fungsi untuk mengekstrak address kolom
    @staticmethod
    def parse_address(addr):
        dict_addr = usaddress.tag(addr)
        return dict_addr

    @staticmethod
    def state_transform(x):
        # Mengubah code state dari us ke nama lengkapnya
        us_state_to_abbrev = {
            "Alabama": "AL",
            "Alaska": "AK",
            "Arizona": "AZ",
            "Arkansas": "AR",
            "California": "CA",
            "Colorado": "CO",
            "Connecticut": "CT",
            "Delaware": "DE",
            "Florida": "FL",
            "Georgia": "GA",
            "Hawaii": "HI",
            "Idaho": "ID",
            "Illinois": "IL",
            "Indiana": "IN",
            "Iowa": "IA",
            "Kansas": "KS",
            "Kentucky": "KY",
            "Louisiana": "LA",
            "Maine": "ME",
            "Maryland": "MD",
            "Massachusetts": "MA",
            "Michigan": "MI",
            "Minnesota": "MN",
            "Mississippi": "MS",
            "Missouri": "MO",
            "Montana": "MT",
            "Nebraska": "NE",
            "Nevada": "NV",
            "New Hampshire": "NH",
            "New Jersey": "NJ",
            "New Mexico": "NM",
            "New York": "NY",
            "North Carolina": "NC",
            "North Dakota": "ND",
            "Ohio": "OH",
            "Oklahoma": "OK",
            "Oregon": "OR",
            "Pennsylvania": "PA",
            "Rhode Island": "RI",
            "South Carolina": "SC",
            "South Dakota": "SD",
            "Tennessee": "TN",
            "Texas": "TX",
            "Utah": "UT",
            "Vermont": "VT",
            "Virginia": "VA",
            "Washington": "WA",
            "West Virginia": "WV",
            "Wisconsin": "WI",
            "Wyoming": "WY",
            "District of Columbia": "DC",
            "American Samoa": "AS",
            "Guam": "GU",
            "Northern Mariana Islands": "MP",
            "Puerto Rico": "PR",
            "United States Minor Outlying Islands": "UM",
            "Virgin Islands, U.S.": "VI",
        }
        us_reverse = {value: key for key, value in us_state_to_abbrev.items()}
        return us_reverse[x]

    def combining(self):
        column_mapping = {
            "Amount": "amount",
            "transactionAmount": "amount",
            "amt": "amount",
            "Location": "location",
            "StateName": "location",
            "state": "location",
            "IsFraud": "is_fraud",
            "Fraud": "is_fraud",
            "Fraudulent": "is_fraud",
        }
        # Rename columns in each dataframe inside the dictionary
        for table_name, df_list in self.dataframes.items():
            for i, df in enumerate(df_list):
                df_list[i] = df.rename(columns=column_mapping)

        keep_cols = ["amount", "location", "is_fraud"]

        list_col = []
        dfs_append = [
            list_col.append(df[0].columns.tolist()) for df in self.dataframes.values()
        ]
        print(f"list_col: {list_col}")
        for i in list_col:
            print(f"i : {i}")
            if "is_fraud" not in i:
                print(f"removed")
                keep_cols.remove("is_fraud")
                continue

        # Flatten all dataframes into a single list
        all_dfs = [df[keep_cols] for dfs in self.dataframes.values() for df in dfs]

        df_concat = pd.concat(all_dfs, ignore_index=True)
        print(f"hasil omegad dan len concat: {len(df_concat)}")
        # Print length of each dataframe
        for table_name, df_list in self.dataframes.items():
            for i, df in enumerate(df_list):
                print(f"{table_name}[{i}] length: {len(df)}")
        print(df_concat.head())
        return df_concat

    def feature_extraction(self):
        for name, df in self.dataframes.items():
            df = df[0]
            list_col = df.columns.tolist()
            if "customerBillingAddress" in list_col:
                df_new = (
                    df["customerBillingAddress"]
                    .apply(self.__class__.parse_address)
                    .apply(pd.Series)[0]
                    .apply(pd.Series)
                )
                df = pd.concat([df, df_new], axis=1)
                self.dataframes[name] = [df]

            common_items = list(set(["StateName", "state"]).intersection(list_col))
            if common_items:
                df[common_items] = df[common_items].map(self.__class__.state_transform)
                # df3['state'] = df3['state'].map(us_reverse)
                self.dataframes[name] = [df]

    def validation(self):
        # Ngecek length dari dataframes
        for name, df in self.dataframes.items():
            if len(df) > 1:
                return "Pastikan dalam table cuma ada 1 dataframe, kalau belum join kolom atau hapus table"

        # Ngecek ada yg missing apa ga
        required = ["amount", "location"]

        for name, df in self.dataframes.items():
            for col in required:
                if col not in df.columns:
                    return "Missing column: " + col
        return True

    def preprocessing(self):
        valid = self.validation()
        if valid != True:
            return valid

        self.feature_extraction()

        df_concat = self.combining()

        return df_concat
