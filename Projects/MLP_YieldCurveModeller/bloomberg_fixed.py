# -*- coding: utf-8 -*-
"""
Created on Wednesday Jan 9 00:00:00 2019

@author: RR68975
"""

import blpapi
import pandas as pd
import datetime as dt

class Bloomberg:
    def __init__ (self, host='localhost', port=8194, connect=True, debug=False):
        self.debug = debug
        self.active = False
        if connect:
            self.connect(host, port)

    def connect (self, host, port):
        if not self.active:
            self._debug("OPENING Bberg connexion")
            sessionOptions = blpapi.SessionOptions()
            sessionOptions.setServerHost(host)
            sessionOptions.setServerPort(port)
            self.session = blpapi.Session(sessionOptions)
            self.session.start()
            self.session.openService('//BLP/refdata')
            self.refDataService = self.session.getService('//BLP/refdata')
            self.active = True
            self._debug("OPENING Bberg connexion - DONE")

    def close (self):
        if self.active:
            self._debug("CLOSING Bberg connexion")
            self.session.stop()
            self.active = False

    def referenceRequest (self, securities, fields, overrides = {}, **kwargs):
        if len(securities)==0 or len(fields)==0:
            return pd.DataFrame()

        self._debug("REFERENCE REQUEST - Sending ...")
        timerequest = dt.datetime.now()
        timerequest = timerequest.strftime("%Y%m%d - %H:%M:%S")
        response, data = self.sendRequest('ReferenceData', securities, fields, overrides, kwargs)
        self._debug("REFERENCE REQUEST - Received ...")
        data['bbergvalue'] = None
        data['status'] = None
        self._debug("REFERENCE REQUEST - Processing ...")
        bbg_data = []
        for msg in response:
            #We check if there is any info at all
            if msg.hasElement('securityData'):
                securitiesData = msg.getElement('securityData')
                for securityData in securitiesData.values():
                    security = securityData.getElementValue('security')
                    # We check for errors at the security level
                    if securityData.hasElement('securityError'):
                        # We have an incorrect security
                        securityError = securityData.getElement('securityError')
                        data.loc[data['bbergsymbol'] == security, 'bbergvalue'] = securityError.getElementValue('subcategory')
                        data.loc[data['bbergsymbol'] == security, 'status'] = "ERROR - {0} - {1}".format(securityError.getElementValue('message'), timerequest)
                    else:
                        # We have a correct security
                        # We get the field data
                        fieldData = securityData.getElement('fieldData')
                        # We iterate for each field data
                        for each in fieldData.elements():
                            bbergfield = str(each.name())
                            if each.isArray():
                                data_bulk = pd.DataFrame()
                                # We have bulk data
                                for BulkArray in each.values():
                                    values = []
                                    columns = []
                                    for Elements in BulkArray.elements():
                                        values += [Elements.getValue()]
                                        columns += [str(Elements.name())]
                                    data_bulk = pd.concat([data_bulk, pd.DataFrame([values], columns = columns)], ignore_index = True)
                                value = str(data_bulk.to_dict())
                            else:
                                value = each.getValue()

                            bbg_data.append([security, bbergfield, value, "DONE - " + timerequest])

                        # We check if we have incorrect fields
                        if securityData.hasElement('fieldExceptions'):
                            fieldExceptions = securityData.getElement('fieldExceptions')
                            # We iterate on each wrong field
                            for fieldException in fieldExceptions.values():
                                # We iterate on each wrong field
                                bbergfield = fieldException.getElementValue("fieldId")
                                errorInfo = fieldException.getElement("errorInfo")
                                # bbergvalue = errorInfo.getElementValue('subcategory')
                                bbergvalue = "ERROR - " + errorInfo.getElementValue("message")
                                status = "DONE - " + timerequest
                                bbg_data.append([security, bbergfield, bbergvalue, status])

        bbg_data = pd.DataFrame(bbg_data, columns=data.columns)
        data = pd.concat([data, bbg_data], ignore_index=True).drop_duplicates(['bbergsymbol','bbergfield'], keep='last')
        data.loc[data['status'].isnull(), 'status'] = "ERROR - Field not returned - {0}".format(timerequest)
        data = data.reset_index(drop = True)

        self._debug("REFERENCE REQUEST - DONE")
        return data

    def historicalRequest (self, securities, fields, startdate, enddate, overrides = {}, **kwargs):
        """


        """
        if len(securities)==0 or len(fields)==0:
            return pd.DataFrame()

        if type(startdate) is int:
            startdate = str(startdate)
        if type(enddate) is int:
            enddate = str(enddate)

        if type(startdate) is not str:
            startdate = startdate.strftime("%Y%m%d")
        if type(enddate) is not str:
            enddate = enddate.strftime("%Y%m%d")

        defaults = {'startDate'       : startdate,
            'endDate'                 : enddate,
            'periodicityAdjustment'   : 'ACTUAL', # 'CALENDAR', 'FISCAL'
            'periodicitySelection'    : 'DAILY', # 'WEEKLY', 'MONTHLY', 'QUARTERLY', 'SEMI_ANNUALLY', 'YEARLY'
            # 'currency'
            'pricingOption'           : 'PRICING_OPTION_PRICE', # 'PRICING_OPTION_YIELD'
            'nonTradingDayFillOption' : 'NON_TRADING_WEEKDAYS', #'ACTIVE_DAYS_ONLY', 'NON_TRADING_WEEKDAYS', 'ALL_CALENDAR_DAYS'
            'nonTradingDayFillMethod' : 'PREVIOUS_VALUE', # 'NIL_VALUE'
            # 'maxDataPoints'
            'adjustmentNormal'        : False,
            'adjustmentAbnormal'      : False,
            'adjustmentSplit'         : True,
            'adjustmentFollowDPDF'    : False}
        defaults.update(kwargs)

        timerequest = dt.datetime.now()
        timerequest = timerequest.strftime("%Y%m%d - %H:%M:%S")
        self._debug("HISTORICAL REQUEST - Sending ...")
        response, data = self.sendRequest('HistoricalData', securities, fields, overrides, defaults)
        self._debug("HISTORICAL REQUEST - Received ...")
        data['bbergdate'] = None
        data['bbergvalue'] = None
        data['status'] = None

        bbg_data = []
        self._debug("HISTORICAL REQUEST - Processing ...")
        for msg in response:
            #We check if there is any info at all
            if msg.hasElement('securityData'):
                securityData = msg.getElement('securityData')
                security = securityData.getElementValue('security')

                # We check for errors at the security level
                if securityData.hasElement('securityError'):
                    # We have an incorrect security
                    securityError = securityData.getElement('securityError')
                    data.loc[data['bbergsymbol'] == security, 'bbergvalue'] = securityError.getElementValue('subcategory')
                    data.loc[data['bbergsymbol'] == security, 'status'] = "ERROR - {0} - {1}".format(securityError.getElementValue('message'), timerequest)

                elif securityData.hasElement('fieldData'):
                    # We have a correct security
                    fieldData = securityData.getElement('fieldData')

                    for fld in fieldData.values():
                        bbergsymbol = security
                        bbergdate = fld.getElementAsDatetime('date')
                        for v in [fld.getElement(i) for i in range(fld.numElements()) if fld.getElement(i).name() != 'date']:
                            bbergfield = str(v.name())
                            bbergvalue = v.getValue()
                            status = "DONE - " + timerequest
                            bbg_data.append([bbergsymbol, bbergfield, bbergdate, bbergvalue, status])

                    # We check if we have incorrect fields
                    if securityData.hasElement('fieldExceptions'):
                        fieldExceptions = securityData.getElement('fieldExceptions')
                        # We iterate on each wrong field
                        for fieldException in fieldExceptions.values():
                            # We iterate on each wrong field
                            bbergfield = fieldException.getElementValue("fieldId")
                            errorInfo = fieldException.getElement("errorInfo")
                            bbergvalue = errorInfo.getElementValue('subcategory')
                            status = "ERROR - {0} - {1}".format(errorInfo.getElementValue("message"), timerequest)
                            bbg_data.append([bbergsymbol, bbergfield, bbergdate, bbergvalue, status])


        bbg_data = pd.DataFrame(bbg_data, columns = data.columns)
        data.loc[~data['bbergsymbol'].isin(bbg_data['bbergsymbol'].unique()), 'status'] = "ERROR - Symbol not returned - {0}".format(timerequest)
        data = data[data['status'].notnull()]
        data = pd.concat([data, bbg_data], ignore_index = True)
        data['bbergdate'] = pd.to_datetime(data['bbergdate'])

        self._debug("HISTORICAL REQUEST - DONE")
        return data

    def intradaybarRequest (self, securities, startdate, enddate, overrides = {}, **kwargs):
        """


        """
        if len(securities)==0:
            return pd.DataFrame()

        fields = ''

        defaults = {'startDateTime'   : startdate,
            'endDateTime'             : enddate,
            'eventType'               :'TRADE', # 'BID', 'ASK', 'BID_BEST', 'ASK_BEST', 'BEST_BID', 'BEST_ASK'
            'interval'                : 60, # 1...1440
            'gapFillInitialBar'       : True,
            'adjustmentNormal'        : False,
            'adjustmentAbnormal'      : False,
            'adjustmentSplit'         : True,
            'adjustmentFollowDPDF'    : False}
        defaults.update(kwargs)

        timerequest = dt.datetime.now()
        timerequest = timerequest.strftime("%Y%m%d - %H:%M:%S")

        if type(securities) == str:
            securities = [securities]

        bbg_data = []
        for security in securities:
            self._debug("INTRADAY BAR REQUEST - Sending ...")
            response, data = self.sendRequest('IntradayBar', security, fields, overrides, defaults)
            self._debug("INTRADAY BAR REQUEST REQUEST - Received ...")
            data['bbergdate'] = None
            data['bbergvalue'] = None
            data['status'] = None


            self._debug("INTRADAY BAR REQUEST REQUEST - Processing ...")
            for msg in response:
                #We check if there is any info at all
                if msg.hasElement('barData'):
                    bbergsymbol = security

                    # We check for errors at the security level
                    if msg.hasElement('responseError'):
                        # We have an incorrect security
                        securityError = msg.getElement('responseError')
                        data.loc[data['bbergsymbol'] == bbergsymbol, 'bbergvalue'] = securityError.getElementValue('subcategory')
                        data.loc[data['bbergsymbol'] == bbergsymbol, 'status'] = "ERROR - {0} - {1}".format(securityError.getElementValue('message'), timerequest)

                    else:
                        barData = msg.getElement('barData')
                        barTickData = barData.getElement('barTickData')
                        for bartickData in barTickData.values():
                            bbergdate = bartickData.getElementAsDatetime('time')
                            bbergfields = ['open', 'high', 'low', 'close', 'volume']
                            status = "DONE - " + timerequest
                            for bbergfield in bbergfields:
                                bbergvalue = bartickData.getElementAsFloat(bbergfield)
                                bbg_data.append([bbergsymbol, bbergfield, bbergdate, bbergvalue, status])


        bbg_data = pd.DataFrame(bbg_data, columns = data.columns)
        data.loc[~data['bbergsymbol'].isin(bbg_data['bbergsymbol'].unique()), 'status'] = "ERROR - Symbol not returned - {0}".format(timerequest)
        data = data[data['status'].notnull()]
        data = pd.concat([data, bbg_data], ignore_index = True)

        self._debug("INTRADAY BAR REQUEST REQUEST - DONE")
        return data

    def sendRequest (self, requestType, securities, fields, overrides={}, elements={}):
        """ Prepares and sends a request then blocks until it can return
            the complete response.

            Depending on the complexity of your request, incomplete and/or
            unrelated messages may be returned as part of the response.
        """
        request = self.refDataService.createRequest(requestType + 'Request')

        # Convert security and field to list if single string
        if type(securities) == str:
            securities = [securities]
        if type(fields) == str:
            fields = [fields]

        if requestType == 'IntradayBar':
            request.set("security", securities[0])
        else:
            for s in securities:
                request.getElement("securities").appendValue(s)
            for f in fields:
                request.getElement("fields").appendValue(f)
            # add overrides
            req_override = request.getElement("overrides")
            req_overrides = []
            for each in overrides:
                req_overrides.append(req_override.appendElement())
                req_overrides[-1].setElement("fieldId", each)
                req_overrides[-1].setElement("value", overrides[each])
        for k, v in elements.items():
            if type(v) == dt:
                if requestType == 'HistoricalData':
                    v = v.strftime('%Y%m%d')

            request.set(k, v)

        self.session.sendRequest(request)

        response = []
        while True:
            event = self.session.nextEvent(10)
            for msg in event:

                if msg.messageType() == requestType + 'Response':
                    response.append(msg)

            if event.eventType() == blpapi.Event.RESPONSE:
                break

        # Create the Data DataFrame
        data = []
        for security in securities:
            for field in fields:
                data.append([security, field])
        data = pd.DataFrame(data, columns = ['bbergsymbol', 'bbergfield'])

        return response, data

    def referenceRequest2(self, df, securities, fields, overrides = {}, **kwargs):
        data = pd.DataFrame()

        # A reference Request but checking for existing fields
        df = df[(df['status'].str[:4].str.upper()=='DONE')]

        bbergsymbol = pd.DataFrame({'bbergsymbol':securities, 'merge':1})
        bbergfield = pd.DataFrame({'bbergfield':fields, 'merge':1})
        request = bbergsymbol.merge(bbergfield, how='left')
        request = request.drop_duplicates()

        request = request.merge(df, how='left', on=['bbergsymbol', 'bbergfield'])
        request = request[request['status'].str[:4] != "DONE"]

        for field in request['bbergfield'].unique():
            field_securities = request[request['bbergfield'] == field]['bbergsymbol'].unique()
            data = pd.concat([data, self.referenceRequest(field_securities, field, overrides, **kwargs)], ignore_index=True)

        return data

    def historicalRequest2(self, df, securities, fields, datearray, overrides = {}, **kwargs):
        data = pd.DataFrame()

        # A reference Request but checking for existing fields
        df = df[(df['status'].str[:4].str.upper()=='DONE')]
        df['bbergdate'] = pd.to_datetime(df['bbergdate']).copy()

        bbergsymbol = pd.DataFrame({'bbergsymbol':securities, 'merge':1})
        bbergfield = pd.DataFrame({'bbergfield':fields, 'merge':1})
        bbergdate = pd.DataFrame({'bbergdate': datearray, 'merge':1})
        bbergdate['bbergdate'] = pd.to_datetime(bbergdate['bbergdate'])
        request = bbergsymbol.merge(bbergfield, how='left')
        request = request.merge(bbergdate, how='left')
        request = request.drop_duplicates()

        request = request.merge(df, how='left', on=['bbergsymbol', 'bbergfield', 'bbergdate'])
        request = request[request['status'].str[:4] != "DONE"]

        for date in request['bbergdate'].dt.date.unique():
            date_securitiesandfields = request[request['bbergdate'] == date]
            for field in date_securitiesandfields['bbergfield'].unique():
                field_securities = date_securitiesandfields[date_securitiesandfields['bbergfield'] == field]['bbergsymbol'].unique()
                data = pd.concat([data, self.historicalRequest(field_securities, field, date, date, overrides, **kwargs)], ignore_index=True)
        return data

    def __enter__ (self):
        self.connect()
        return self

    def __exit__ (self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__ (self):
        self.close()

    def _debug (self, msg):
        if self.debug:
            print("{0} - BBG DEBUG: {1}".format(dt.datetime.now().strftime("%Y%m%d %H:%M:%S"), msg))