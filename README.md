from zeep import Client

# Creating SOAP clients
soap = Client(wsdl="http://localhost:8080/connect/wsdl/SecurityProvider?wsdl")
soap_journal = Client(wsdl="http://localhost:8080/connect/wsdl/ComponentExecutor?wsdl")

# Authenticating
auth_response = soap.service.Authenticate(name="AOK", password="")
vaucher = auth_response.get('vaucher')
print(vaucher)

# Making SOAP call for Journal
params_journal = {
    "authentication": vaucher,
    "licensing": "",
    "component": "Journal",
    "method": "Import",
    "group": "",
    "payload": '''<SSC>
                    <SunSystemsContext>
                        <BusinessUnit>CEA</BusinessUnit>
                        <BudgetCode>A</BudgetCode>
                    </SunSystemsContext>
                    <MethodContext>
                        <LedgerPostingParameters>
                            <JournalType>JV</JournalType>
                            <PostingType>2</PostingType>
                            <PostProvisional>N</PostProvisional>
                            <PostToHold>N</PostToHold>
                            <BalancingOptions>T2</BalancingOptions>
                            <SuspenseAccount>338100</SuspenseAccount>
                            <TransactionAmountAccount>338100</TransactionAmountAccount>
                            <ReportingAccount>338100</ReportingAccount>
                            <SupressSubstitutedMessages>N</SupressSubstitutedMessages>
                            <ReportErrorsOnly>Y</ReportErrorsOnly>
                        </LedgerPostingParameters>
                    </MethodContext>
                    <Payload>
                        <Ledger>
                            <Line>
                                <TransactionReference>651C</TransactionReference>
                                <AccountingPeriod>0052017</AccountingPeriod>
                                <TransactionDate>07052017</TransactionDate>
                                <AccountCode>ERROJAB01</AccountCode>
                                <AnalysisCode2/>
                                <AnalysisCode3>10</AnalysisCode3>
                                <AnalysisCode4/>
                                <AnalysisCode5/>
                                <AnalysisCode6/>
                                <AnalysisCode7/>
                                <AnalysisCode8/>
                                <AnalysisCode9/>
                                <AnalysisCode10/>
                                <Description>GONZALEZ ALCUDIA HUMBERTO</Description>
                                <Value4Amount>3500</Value4Amount>
                                <DebitCredit>D</DebitCredit>
                                <Value4CurrencyCode>MXP1</Value4CurrencyCode>
                                <DueDate>07052017</DueDate>
                            </Line>
                            <Line>
                                <TransactionReference>651C</TransactionReference>
                                <AccountingPeriod>0052017</AccountingPeriod>
                                <TransactionDate>07052017</TransactionDate>
                                <AccountCode>ERROJAB01</AccountCode>
                                <AnalysisCode2/>
                                <AnalysisCode3>10</AnalysisCode3>
                                <AnalysisCode4/>
                                <AnalysisCode5/>
                                <AnalysisCode6/>
                                <AnalysisCode7/>
                                <AnalysisCode8/>
                                <AnalysisCode9/>
                                <AnalysisCode10/>
                                <Description>GONZALEZ ALCUDIA HUMBERTO</Description>
                                <Value4Amount>3500</Value4Amount>
                                <DebitCredit>C</DebitCredit>
                                <Value4CurrencyCode>MXP1</Value4CurrencyCode>
                                <DueDate>07052017</DueDate>
                            </Line>
                        </Ledger>
                    </Payload>
                </SSC>'''
}

# Making SOAP call for Journal Execution
journal_response = soap_journal.service.Execute(**params_journal)
diario = journal_response.get('diario')
print(diario)
