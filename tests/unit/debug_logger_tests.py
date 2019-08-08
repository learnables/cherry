#!/usr/bin/env python3

import unittest
import cherry as ch
import cherry.envs as envs
import logging
import os


class TestDebugLogger(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_propagation(self):
        with self.assertLogs('cherry', level='INFO') as context:
            logging.getLogger('cherry').critical('Rainier cherries rule.')
            ch.debug.logger.error('Yann LeCun is notably ambivalent towards cherries.')
            logging.getLogger('cherry').info('The average cherry tree yields ~7,000 cherries annually.')
            ch.debug.logger.debug('Washington, Oregon and California combine to make 94% of the United States sweet cherries.')
            logging.basicConfig(stream=open(os.devnull,'w'))
            logging.critical('Cherries are typically harvested via a mechanical tree shaker and tarp.')
            logging.warning('The word cherry is thought to be derived from the name of the Greek city Cerasus.')
            logging.debug('The record for cherry pit spitting distance (w/ no running start) is 93 feet, held by Brian Krause')
        
        self.assertEqual(context.output, ['CRITICAL:cherry:Rainier cherries rule.',
                                          'ERROR:cherry:Yann LeCun is notably ambivalent towards cherries.',
                                          'INFO:cherry:The average cherry tree yields ~7,000 cherries annually.'])


if __name__ == '__main__':
    unittest.main()
